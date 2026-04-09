#ifndef GNN_ENCODER_HPP
#define GNN_ENCODER_HPP

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace s_graphs {

/**
 * @brief Pure C++/Eigen implementation of the GAT-based GNN encoder.
 *
 * Architecture:
 *   Linear(537,128) → GATConv(128,128,heads=4) → GATConv(128,128,heads=4)
 *   → global_mean_pool → Linear(128,128) → L2 normalize
 *
 * Weights are loaded from raw binary files exported by export_weights.py.
 * No libtorch / no Python / no external ML dependencies.
 */
class GNNEncoder {
 public:
  GNNEncoder() = default;

  bool load(const std::string& weights_dir) {
    try {
      // Load config
      std::string config_path = weights_dir + "/config.txt";
      std::ifstream cfg(config_path);
      if (!cfg.is_open()) {
        std::cerr << "[GNN_ENCODER] Cannot open config: " << config_path << std::endl;
        return false;
      }
      std::string key;
      while (cfg >> key) {
        int val;
        cfg >> val;
        if (key == "input_dim") input_dim_ = val;
        else if (key == "hidden_dim") hidden_dim_ = val;
        else if (key == "output_dim") output_dim_ = val;
        else if (key == "heads") heads_ = val;
      }
      head_dim_ = hidden_dim_ / heads_;

      // Load weights
      load_matrix(weights_dir + "/node_proj_weight.bin", node_proj_W_);
      load_vector(weights_dir + "/node_proj_bias.bin", node_proj_b_);
      load_matrix(weights_dir + "/conv1_W.bin", conv1_W_);
      load_matrix(weights_dir + "/conv1_att_src.bin", conv1_att_src_);
      load_matrix(weights_dir + "/conv1_att_dst.bin", conv1_att_dst_);
      load_matrix(weights_dir + "/conv2_W.bin", conv2_W_);
      load_matrix(weights_dir + "/conv2_att_src.bin", conv2_att_src_);
      load_matrix(weights_dir + "/conv2_att_dst.bin", conv2_att_dst_);
      load_matrix(weights_dir + "/out_proj_weight.bin", out_proj_W_);
      load_vector(weights_dir + "/out_proj_bias.bin", out_proj_b_);

      loaded_ = true;
      std::cout << "[GNN_ENCODER] Loaded from " << weights_dir
                << " (dim=" << input_dim_ << "→" << hidden_dim_
                << "→" << output_dim_ << ", heads=" << heads_ << ")" << std::endl;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[GNN_ENCODER] Load failed: " << e.what() << std::endl;
      loaded_ = false;
      return false;
    }
  }

  bool is_loaded() const { return loaded_; }

  /**
   * @brief Encode a subgraph → 128-dim L2-normalized embedding.
   */
  std::vector<float> encode(
      const std::vector<float>& node_features,
      int num_nodes,
      int feat_dim,
      const std::vector<int>& edge_src,
      const std::vector<int>& edge_dst) {
    if (!loaded_ || num_nodes == 0) {
      return std::vector<float>(output_dim_, 0.0f);
    }

    // Map flat vector → Eigen matrix [N x feat_dim]
    Eigen::MatrixXf X = Eigen::Map<const Eigen::MatrixXf>(
        node_features.data(), feat_dim, num_nodes).transpose();

    // 1. Linear projection: X = ReLU(X @ W^T + b)
    X = ((X * node_proj_W_.transpose()).rowwise() + node_proj_b_.transpose())
            .cwiseMax(0.0f);

    // 2. GAT layer 1 (with self-loops)
    X = gat_forward(X, edge_src, edge_dst, conv1_W_, conv1_att_src_, conv1_att_dst_);
    X = X.cwiseMax(0.0f);  // ReLU

    // 3. GAT layer 2 (with self-loops)
    X = gat_forward(X, edge_src, edge_dst, conv2_W_, conv2_att_src_, conv2_att_dst_);
    X = X.cwiseMax(0.0f);  // ReLU

    // 4. Global mean pool → [1 x hidden_dim]
    Eigen::VectorXf pooled = X.colwise().mean();

    // 5. Output projection
    Eigen::VectorXf out = out_proj_W_ * pooled + out_proj_b_;

    // 6. L2 normalize
    float norm = out.norm();
    if (norm > 1e-8f) out /= norm;

    return std::vector<float>(out.data(), out.data() + out.size());
  }

 private:
  // ── GAT layer forward ──
  Eigen::MatrixXf gat_forward(
      const Eigen::MatrixXf& X,             // [N, in_dim]
      const std::vector<int>& edge_src,
      const std::vector<int>& edge_dst,
      const Eigen::MatrixXf& W,             // [heads*head_dim, in_dim]
      const Eigen::MatrixXf& att_src,        // [heads, head_dim]
      const Eigen::MatrixXf& att_dst) {      // [heads, head_dim]

    int N = static_cast<int>(X.rows());

    // Project: H = X @ W^T → [N, heads*head_dim]
    Eigen::MatrixXf H = X * W.transpose();

    // Reshape to [N, heads, head_dim] — stored as [N*heads, head_dim]
    // For attention, compute per-head scores

    // Allocate output
    Eigen::MatrixXf out = Eigen::MatrixXf::Zero(N, heads_ * head_dim_);

    for (int h = 0; h < heads_; h++) {
      // Extract head h: [N, head_dim]
      Eigen::MatrixXf Hh = H.middleCols(h * head_dim_, head_dim_);

      // Source attention scores: e_src[i] = Hh[i] @ att_src[h]
      Eigen::VectorXf e_src_scores = Hh * att_src.row(h).transpose();
      Eigen::VectorXf e_dst_scores = Hh * att_dst.row(h).transpose();

      // Aggregate with attention (including self-loops)
      // For each node j, collect incoming messages with softmax attention
      Eigen::MatrixXf node_out = Eigen::MatrixXf::Zero(N, head_dim_);
      Eigen::VectorXf denom = Eigen::VectorXf::Zero(N);
      Eigen::VectorXf max_score = Eigen::VectorXf::Constant(N, -1e9f);

      // Self-loops + edges
      int E = static_cast<int>(edge_src.size());

      // First pass: find max score per destination (for numerical stability)
      for (int j = 0; j < N; j++) {
        float self_score = leaky_relu(e_src_scores(j) + e_dst_scores(j));
        max_score(j) = self_score;
      }
      for (int e = 0; e < E; e++) {
        int s = edge_src[e], d = edge_dst[e];
        float score = leaky_relu(e_src_scores(s) + e_dst_scores(d));
        if (score > max_score(d)) max_score(d) = score;
      }

      // Second pass: compute softmax + weighted sum
      // Self-loop contribution
      for (int j = 0; j < N; j++) {
        float self_score = leaky_relu(e_src_scores(j) + e_dst_scores(j));
        float w = std::exp(self_score - max_score(j));
        denom(j) += w;
        node_out.row(j) += w * Hh.row(j);
      }

      // Edge contributions
      for (int e = 0; e < E; e++) {
        int s = edge_src[e], d = edge_dst[e];
        float score = leaky_relu(e_src_scores(s) + e_dst_scores(d));
        float w = std::exp(score - max_score(d));
        denom(d) += w;
        node_out.row(d) += w * Hh.row(s);
      }

      // Normalize
      for (int j = 0; j < N; j++) {
        if (denom(j) > 1e-8f) {
          node_out.row(j) /= denom(j);
        }
      }

      out.middleCols(h * head_dim_, head_dim_) = node_out;
    }

    return out;
  }

  static float leaky_relu(float x, float slope = 0.2f) {
    return x >= 0.0f ? x : slope * x;
  }

  // ── Binary file loading ──
  void load_matrix(const std::string& path, Eigen::MatrixXf& mat) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    int ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    std::vector<int> shape(ndim);
    for (int i = 0; i < ndim; i++) {
      f.read(reinterpret_cast<char*>(&shape[i]), 4);
    }

    int rows = shape[0];
    int cols = (ndim >= 2) ? shape[1] : 1;

    std::vector<float> data(rows * cols);
    f.read(reinterpret_cast<char*>(data.data()), rows * cols * 4);

    // NumPy stores row-major, Eigen default is column-major
    mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data.data(), rows, cols);
  }

  void load_vector(const std::string& path, Eigen::VectorXf& vec) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    int ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    int size;
    f.read(reinterpret_cast<char*>(&size), 4);

    std::vector<float> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size * 4);

    vec = Eigen::Map<Eigen::VectorXf>(data.data(), size);
  }

  // ── Model weights ──
  Eigen::MatrixXf node_proj_W_;   // [128, 517]
  Eigen::VectorXf node_proj_b_;   // [128]
  Eigen::MatrixXf conv1_W_, conv2_W_;           // [128, 128]
  Eigen::MatrixXf conv1_att_src_, conv1_att_dst_; // [4, 32]
  Eigen::MatrixXf conv2_att_src_, conv2_att_dst_; // [4, 32]
  Eigen::MatrixXf out_proj_W_;    // [128, 128]
  Eigen::VectorXf out_proj_b_;    // [128]

  int input_dim_ = 537, hidden_dim_ = 128, output_dim_ = 128;
  int heads_ = 4, head_dim_ = 32;
  bool loaded_ = false;
};

}  // namespace s_graphs

#endif  // GNN_ENCODER_HPP
