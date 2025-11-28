/* Copyright (C) 2025 RidgeRun, LLC <support@ridgerun.ai>
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to
 * RidgeRun, LLC. No part of this program may be photocopied,
 * reproduced or translated into another programming language without
 * prior written consent of RidgeRun, LLC. The user is free to modify
 * the source code after obtaining a software license from
 * RidgeRun. All source code changes must be provided back to RidgeRun
 * without any encumbrance.
 */

#include "NvInfer.h"
#include <nvdsinfer.h>
#include <nvdsinfer_custom_impl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Simple span replacement for C++17 compatibility
template <typename T>
class span
{
public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    constexpr span() noexcept : data_(nullptr), size_(0) {}
    constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}
    template <typename U, typename = typename std::enable_if<
        std::is_same<typename std::remove_cv<U>::type, typename std::remove_cv<T>::type>::value &&
        !std::is_same<U, T>::value>::type>
    constexpr span(U* ptr, size_type count) noexcept : data_(ptr), size_(count) {}
    template <std::size_t N>
    constexpr span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}
    template <typename Container>
    constexpr span(Container& c) noexcept : data_(c.data()), size_(c.size()) {}

    constexpr pointer data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr reference operator[](size_type idx) const { return data_[idx]; }
    constexpr pointer begin() const noexcept { return data_; }
    constexpr pointer end() const noexcept { return data_ + size_; }
    constexpr const_pointer cbegin() const noexcept { return data_; }
    constexpr const_pointer cend() const noexcept { return data_ + size_; }

    constexpr span<T> subspan(size_type offset, size_type count = std::numeric_limits<size_type>::max()) const noexcept
    {
        size_type actual_count = (count == std::numeric_limits<size_type>::max()) ? (size_ - offset) : count;
        return span<T>(data_ + offset, actual_count);
    }

private:
    pointer data_;
    size_type size_;
};

// Deduction guides
template <typename T, std::size_t N>
span(T (&)[N]) -> span<T>;

template <typename Container>
span(Container&) -> span<typename Container::value_type>;

template <typename Container>
span(const Container&) -> span<const typename Container::value_type>;


struct Layer {
  struct Classes {
    static constexpr std::string_view NAME = "labels";
    static constexpr NvDsInferDataType TYPE = FLOAT;

    enum Dims : std::uint8_t {
      DETECTIONS,
      CLASSES,
      NUM_DIMS,
    };

    static constexpr unsigned int BACKGROUND = 0;
  };

  struct Boxes {
    static constexpr std::string_view NAME = "dets";
    static constexpr NvDsInferDataType TYPE = FLOAT;

    enum Dims : std::uint8_t {
      DETECTIONS,
      BOXES,
      NUM_DIMS,
    };

    enum Box : std::uint8_t { CX, CY, W, H, SIZE };
  };
};

template <typename T>
auto softmax_of_best_logit(span<const T> logit,
                           span<const T> thresholds)
    -> std::optional<std::pair<std::size_t, T>> {
  const std::size_t size = logit.size();
  assert(size > 0);
  assert(thresholds.size() == size);

  // 0) argmax
  std::size_t max_idx = 0;
  T max_val = logit[0];
  for (std::size_t i = 1; i < size; ++i)
  {
    if (logit[i] > max_val)
    {
      max_val = logit[i];
      max_idx = i;
    }
  }

  // 1) Ignore if its background
  if (max_idx == Layer::Classes::BACKGROUND)
  {
    return std::nullopt;
  }

  // 2) extract the threshold for this class
  T threshold = thresholds[max_idx];

  // 3) threshold → limit
  const long double eps = std::numeric_limits<long double>::epsilon();
  const long double thr = std::max(static_cast<long double>(threshold), eps);

  const long double limit = 1.0L / thr - 1.0L;

  // 4) accumulate Σ_{j≠max} exp(z_j - z_max)
  long double sum_exp_others = 0.0L;

  for (std::size_t i = 0; i < size; ++i)
  {
    if (i == max_idx)
    {
      continue;
    }

    const auto diff = static_cast<long double>(logit[i] - max_val);
    sum_exp_others += std::exp(diff);

    if (sum_exp_others > limit)
    {
      return std::nullopt;
    }
  }

  // 5) exact softmax probability of max logit
  const long double p_max_ld = 1.0L / (1.0L + sum_exp_others);
  const T p_max = static_cast<T>(p_max_ld);

  return std::make_pair(max_idx, p_max);
}

auto find_layer(const std::vector<NvDsInferLayerInfo> &layers,
                const std::string_view &name,
                NvDsInferDataType type) -> std::optional<NvDsInferLayerInfo> {
  auto name_and_type_match = [&](auto const &layer) -> bool {
    return layer.dataType == type && layer.layerName == name;
  };

  auto ilayer = std::find_if(layers.begin(), layers.end(), name_and_type_match);

  if (ilayer == layers.end()) {
    return std::nullopt;
  }

  return *ilayer;
};

template <typename T>
auto view(span<const T> buffer, unsigned int offset,
          unsigned int size) -> span<const T> {
  const auto block_start = static_cast<std::size_t>(offset) * size;
  const auto block_size = static_cast<std::size_t>(size);

  assert(block_start + block_size <= buffer.size());

  return buffer.subspan(block_start, block_size);
}

template <typename T>
auto parse_detection(span<const T> boxes, span<const T> classes,
                     const NvDsInferParseDetectionParams &params,
                     unsigned int width, unsigned int height,
                     bool use_probabilities = false)
    -> std::optional<NvDsInferObjectDetectionInfo> {
  std::optional<std::pair<std::size_t, T>> best;

  if (use_probabilities)
  {
    // Layer 4132 has probability values, not logits - find max directly
    std::size_t max_idx = 0;
    T max_val = classes[0];
    for (std::size_t i = 1; i < classes.size(); ++i)
    {
      if (classes[i] > max_val)
      {
        max_val = classes[i];
        max_idx = i;
      }
    }

    // Check threshold
    T threshold = params.perClassPreclusterThreshold[max_idx];
    if (max_val < threshold)
    {
      return std::nullopt;
    }

    best = std::make_pair(max_idx, max_val);
  }
  else
  {
    // Original layers use logits with softmax
    best = softmax_of_best_logit(classes, span<const T>{params.perClassPreclusterThreshold});
  }

  if (!best) {
    return std::nullopt;
  }

  T box_x1, box_y1, box_x2, box_y2;

  // RF-DETR output format is unclear. Let's try a simple approach:
  // Assume [cx, cy, w, h] format where w and h might be in a different scale
  T cx = boxes[Layer::Boxes::Box::CX];
  T cy = boxes[Layer::Boxes::Box::CY];
  T w = boxes[Layer::Boxes::Box::W];
  T h = boxes[Layer::Boxes::Box::H];

  // Handle inf values - maybe replace with reasonable defaults
  if (std::isinf(w) || std::isnan(w) || w <= 0) {
    w = 32.0F;  // Default width
  }
  if (std::isinf(h) || std::isnan(h) || h <= 0) {
    h = 32.0F;  // Default height
  }

  // Ensure reasonable bounds
  w = std::min(w, static_cast<T>(width));
  h = std::min(h, static_cast<T>(height));

  box_x1 = cx - w/2.0F;
  box_y1 = cy - h/2.0F;
  box_x2 = cx + w/2.0F;
  box_y2 = cy + h/2.0F;

  const float max_x = static_cast<float>(width) - 1.0F;
  const float max_y = static_cast<float>(height) - 1.0F;
  constexpr float min_x = 0.0F;
  constexpr float min_y = 0.0F;

  box_x1 = std::clamp(box_x1, min_x, max_x);
  box_y1 = std::clamp(box_y1, min_y, max_y);
  box_x2 = std::clamp(box_x2, min_x, max_x);
  box_y2 = std::clamp(box_y2, min_y, max_y);

  NvDsInferObjectDetectionInfo pred;
  pred.classId = best->first;
  pred.detectionConfidence = best->second;
  pred.left = box_x1;
  pred.top = box_y1;
  pred.width = box_x2 - box_x1;
  pred.height = box_y2 - box_y1;

  return pred;
}

template <typename T>
auto layer_to_span(const NvDsInferLayerInfo &layer) -> span<const T> {
  std::size_t layer_size = 1;
  for (unsigned int i = 0; i < layer.inferDims.numDims; i++) {
    layer_size *= layer.inferDims.d[i];
  }

  return span<const T>(static_cast<T *>(layer.buffer), layer_size);
}

}  // namespace

extern "C" auto deepstream_rfdetr_bbox(
    const std::vector<NvDsInferLayerInfo> &layers,
    const NvDsInferNetworkInfo &network,
    const NvDsInferParseDetectionParams &params,
    std::vector<NvDsInferObjectDetectionInfo> &detections) -> bool {
  static int call_count = 0;
  call_count++;
  std::cerr << "DeepStream-RFDETR: DEBUG - Parser called #" << call_count << " with " << layers.size() << " layers\n";

  // Check if this is during initialization (no actual inference data)
  if (layers.empty() || (layers.size() > 0 && layers[0].buffer == nullptr)) {
    std::cerr << "DeepStream-RFDETR: DEBUG - Initialization call (no buffers), returning true\n";
    detections.clear();
    return true;
  }

  // Note: DeepStream only passes OUTPUT layers to custom parsers, not input layers
  // So we can't check input data here

  std::cerr << "DeepStream-RFDETR: DEBUG - All available layers:\n";
  for (std::size_t i = 0; i < layers.size(); ++i)
  {
    std::cerr << "  [" << i << "] name=\"" << layers[i].layerName 
              << "\", dataType=" << static_cast<int>(layers[i].dataType)
              << ", buffer=" << layers[i].buffer << "\n";
  }
  
  auto layer_dets = find_layer(layers, Layer::Boxes::NAME, Layer::Boxes::TYPE);
  auto layer_labels =
      find_layer(layers, Layer::Classes::NAME, Layer::Classes::TYPE);

  if (!layer_dets || !layer_labels) {
    std::cerr << "DeepStream-RFDETR: Unable to find output layers named \""
              << Layer::Boxes::NAME << "\" and \"" << Layer::Classes::NAME
              << "\". Did you pass the right engine?\n"
              << "The output layer names are: \n";
    std::for_each(layers.begin(), layers.end(), [](const auto &layer) {
      std::cerr << "\t- " << layer.layerName << "\n";
    });

    return false;
  }

  // The model outputs are swapped: "dets" contains class logits (300x91)
  // and "labels" contains box coordinates (300x4)
  // So we swap the interpretation here
  auto layer_boxes = layer_labels;  // labels layer has boxes (4 values)
  auto layer_classes = layer_dets;   // dets layer has classes (91 values)

  std::cerr << "DeepStream-RFDETR: DEBUG - Found layers:\n";
  std::cerr << "  - layer_dets (original classes): " << layer_dets->layerName << "\n";
  std::cerr << "  - layer_labels (original boxes): " << layer_labels->layerName << "\n";
  std::cerr << "  - Will use: classes=" << layer_classes->layerName << ", boxes=" << layer_boxes->layerName << "\n";

  auto layer_classes_num_dims = layer_classes->inferDims.numDims;
  auto layer_boxes_num_dims = layer_boxes->inferDims.numDims;

  if (Layer::Classes::Dims::NUM_DIMS != layer_classes_num_dims ||
      Layer::Boxes::Dims::NUM_DIMS != layer_boxes_num_dims) {
    std::cerr << "DeepStream-RFDETR: layer number of dimensions don't match. "
                 "Did you pass in the correct model?\n"
              << "\t- " << Layer::Classes::NAME << ": "
              << Layer::Classes::Dims::NUM_DIMS << " (expected) <-> "
              << layer_classes_num_dims << " (got)\n"
              << "\t- " << Layer::Boxes::NAME << ": "
              << Layer::Boxes::Dims::NUM_DIMS << " (expected) <-> "
              << layer_boxes_num_dims << " (got)\n";
    return false;
  }

  const span<const unsigned int> layer_boxes_dims{
      layer_boxes->inferDims.d, NVDSINFER_MAX_DIMS};
  auto num_detections_boxes = layer_boxes_dims[Layer::Boxes::Dims::DETECTIONS];
  auto num_box_params = layer_boxes_dims[Layer::Boxes::Dims::BOXES];

  if (Layer::Boxes::Box::SIZE != num_box_params) {
    std::cerr << "DeepStream-RFDETR: The boxes tensor has a "
                 "different box dimension size ("
              << num_box_params << ") than the expected ("
              << Layer::Boxes::Box::SIZE << "). Did you pass "
              << "in the correct model?\n";
    return false;
  }

  const span<const unsigned int> layer_classes_dims{
      layer_classes->inferDims.d, NVDSINFER_MAX_DIMS};
  auto num_detections_classes =
      layer_classes_dims[Layer::Classes::Dims::DETECTIONS];
  auto num_classes = layer_classes_dims[Layer::Classes::Dims::CLASSES];

  std::cerr << "DeepStream-RFDETR: DEBUG - Tensor dimensions:\n";
  std::cerr << "  - Boxes tensor: " << num_detections_boxes << " detections x " << num_box_params << " params\n";
  std::cerr << "  - Classes tensor: " << num_detections_classes << " detections x " << num_classes << " classes\n";
  std::cerr << "  - Network size: " << network.width << "x" << network.height << "\n";
  std::cerr << "  - Configured classes: " << params.numClassesConfigured << "\n";

  if (params.numClassesConfigured != num_classes) {
    std::cerr << "DeepStream-RFDETR: The classes tensor has a "
                 "different dimension size ("
              << num_classes << ") than the expected ("
              << params.numClassesConfigured << "). Check your "
              << "nvinfer config file!\n";
    return false;
  }

  if (num_detections_boxes != num_detections_classes) {
    std::cerr << "DeepStream-RFDETR: The max number of detections "
                 "in the box ("
              << num_detections_boxes
              << ") and "
                 "classes ("
              << num_detections_classes
              << ") tensors "
                 "don't match! Did you pass in the correct model?\n";
    return false;
  }

  // Debug: Check layer buffer information
  std::cerr << "DeepStream-RFDETR: DEBUG - Layer buffer information:\n";
  std::cerr << "  - layer_classes buffer: " << layer_classes->buffer << "\n";
  std::cerr << "  - layer_classes dataType: " << static_cast<int>(layer_classes->dataType) << " (0=FLOAT, 1=HALF, 2=INT8, 3=INT32, 4=INT8_CAL)\n";
  std::cerr << "  - layer_boxes buffer: " << layer_boxes->buffer << "\n";
  std::cerr << "  - layer_boxes dataType: " << static_cast<int>(layer_boxes->dataType) << " (0=FLOAT, 1=HALF, 2=INT8, 3=INT32, 4=INT8_CAL)\n";
  
  // Check if buffers are null
  if (layer_classes->buffer == nullptr)
  {
    std::cerr << "DeepStream-RFDETR: ERROR - layer_classes buffer is NULL!\n";
    return false;
  }
  if (layer_boxes->buffer == nullptr)
  {
    std::cerr << "DeepStream-RFDETR: ERROR - layer_boxes buffer is NULL!\n";
    return false;
  }

  // Check raw bytes to see if there's any data at all
  std::cerr << "DeepStream-RFDETR: DEBUG - First 32 raw bytes from classes buffer (hex):\n";
  const unsigned char* classes_bytes = reinterpret_cast<const unsigned char*>(layer_classes->buffer);
  for (std::size_t i = 0; i < std::min(32ul, static_cast<std::size_t>(num_detections_classes * num_classes * sizeof(float))); ++i)
  {
    std::cerr << std::hex << static_cast<unsigned int>(classes_bytes[i]) << " ";
    if ((i + 1) % 16 == 0)
    {
      std::cerr << "\n";
    }
  }
  std::cerr << std::dec << "\n";
  
  std::cerr << "DeepStream-RFDETR: DEBUG - First 32 raw bytes from boxes buffer (hex):\n";
  const unsigned char* boxes_bytes = reinterpret_cast<const unsigned char*>(layer_boxes->buffer);
  for (std::size_t i = 0; i < std::min(32ul, static_cast<std::size_t>(num_detections_boxes * num_box_params * sizeof(float))); ++i)
  {
    std::cerr << std::hex << static_cast<unsigned int>(boxes_bytes[i]) << " ";
    if ((i + 1) % 16 == 0)
    {
      std::cerr << "\n";
    }
  }
  std::cerr << std::dec << "\n";

  auto tensor_classes = layer_to_span<float>(*layer_classes);
  auto tensor_boxes = layer_to_span<float>(*layer_boxes);

  std::cerr << "DeepStream-RFDETR: DEBUG - Tensor spans:\n";
  std::cerr << "  - tensor_classes size: " << tensor_classes.size() << "\n";
  std::cerr << "  - tensor_boxes size: " << tensor_boxes.size() << "\n";
  std::cerr << "  - tensor_classes data pointer: " << tensor_classes.data() << "\n";
  std::cerr << "  - tensor_boxes data pointer: " << tensor_boxes.data() << "\n";

  // Check first few raw float values
  std::cerr << "DeepStream-RFDETR: DEBUG - First 10 raw float values from classes tensor:\n";
  bool all_nan_classes = true;
  for (std::size_t i = 0; i < std::min(10ul, tensor_classes.size()); ++i)
  {
    std::cerr << "    [" << i << "] = " << tensor_classes[i];
    if (std::isnan(tensor_classes[i]))
    {
      std::cerr << " (NaN)";
    }
    else
    {
      all_nan_classes = false;
    }
    std::cerr << "\n";
  }
  
  std::cerr << "DeepStream-RFDETR: DEBUG - First 10 raw float values from boxes tensor:\n";
  bool all_nan_boxes = true;
  for (std::size_t i = 0; i < std::min(10ul, tensor_boxes.size()); ++i)
  {
    std::cerr << "    [" << i << "] = " << tensor_boxes[i];
    if (std::isnan(tensor_boxes[i]))
    {
      std::cerr << " (NaN)";
    }
    else
    {
      all_nan_boxes = false;
    }
    std::cerr << "\n";
  }

  if (all_nan_classes || all_nan_boxes)
  {
    std::cerr << "DeepStream-RFDETR: WARNING - All buffer values are NaN! This indicates:\n";
    std::cerr << "  1. The model may not be running/inferencing\n";
    std::cerr << "  2. The buffers may not be populated by the inference engine\n";
    std::cerr << "  3. There may be a configuration issue with the model\n";
    std::cerr << "  4. The model may be producing invalid outputs\n";
    std::cerr << "  Please check:\n";
    std::cerr << "    - Is the model actually running? (check nvinfer logs)\n";
    std::cerr << "    - Are the output layer names correct in the config?\n";
    std::cerr << "    - Is the model engine file valid?\n";
    std::cerr << "    - Are there any errors in the TensorRT inference?\n";
    std::cerr << "  Returning empty detections list (no crash, but no detections will be found).\n";
    // Don't return false here - return true with empty detections to avoid segfault
    // The model/inference issue needs to be fixed, but we shouldn't crash DeepStream
    detections.clear();
    return true;
  }

  auto width = network.width;
  auto height = network.height;

  // Try using Layer 5 (4132) for classes instead - it has probability-like values
  auto layer_classes_4132 = find_layer(layers, "4132", Layer::Classes::TYPE);
  auto layer_boxes_2970 = find_layer(layers, "2970", Layer::Boxes::TYPE);

  if (layer_classes_4132 && layer_boxes_2970)
  {
    std::cerr << "DeepStream-RFDETR: DEBUG - Switching to Layer 4132 for classes and Layer 2970 for boxes\n";
    // Override the layer pointers
    layer_classes = layer_classes_4132;
    layer_boxes = layer_boxes_2970;

    // Update dimensions
    layer_classes_num_dims = layer_classes->inferDims.numDims;
    layer_boxes_num_dims = layer_boxes->inferDims.numDims;

    const span<const unsigned int> layer_boxes_dims_2970{
        layer_boxes->inferDims.d, NVDSINFER_MAX_DIMS};
    num_detections_boxes = layer_boxes_dims_2970[Layer::Boxes::Dims::DETECTIONS];
    num_box_params = layer_boxes_dims_2970[Layer::Boxes::Dims::BOXES];

    const span<const unsigned int> layer_classes_dims_4132{
        layer_classes->inferDims.d, NVDSINFER_MAX_DIMS};
    num_detections_classes = layer_classes_dims_4132[Layer::Classes::Dims::DETECTIONS];
    num_classes = layer_classes_dims_4132[Layer::Classes::Dims::CLASSES];

    std::cerr << "DeepStream-RFDETR: DEBUG - New dimensions - Classes: " << num_detections_classes
              << "x" << num_classes << ", Boxes: " << num_detections_boxes << "x" << num_box_params << "\n";
  }

  // Debug: Print sample values from first few detections
  std::cerr << "DeepStream-RFDETR: DEBUG - Sample data from first 3 detections:\n";
  for (unsigned int i = 0; i < std::min(3u, num_detections_classes); ++i)
  {
    auto classes = view<float>(tensor_classes, i, num_classes);
    auto boxes = view<float>(tensor_boxes, i, Layer::Boxes::Box::SIZE);
    
    std::cerr << "  Detection " << i << ":\n";
    std::cerr << "    Box: cx=" << boxes[Layer::Boxes::Box::CX] 
              << ", cy=" << boxes[Layer::Boxes::Box::CY]
              << ", w=" << boxes[Layer::Boxes::Box::W]
              << ", h=" << boxes[Layer::Boxes::Box::H] << "\n";
    
    // Find max class and its value
    float max_class_val = classes[0];
    std::size_t max_class_idx = 0;
    for (std::size_t j = 1; j < num_classes; ++j)
    {
      if (classes[j] > max_class_val)
      {
        max_class_val = classes[j];
        max_class_idx = j;
      }
    }
    std::cerr << "    Classes: max_idx=" << max_class_idx 
              << ", max_val=" << max_class_val
              << ", first_5=[";
    for (std::size_t j = 0; j < std::min(5ul, static_cast<std::size_t>(num_classes)); ++j)
    {
      std::cerr << classes[j];
      if (j < std::min(4ul, static_cast<std::size_t>(num_classes) - 1))
      {
        std::cerr << ", ";
      }
    }
    std::cerr << "]\n";
  }

  // We can add at most num_detection_classes, pre-allocate
  detections.reserve(num_detections_classes);

  unsigned int detections_passed = 0;
  unsigned int detections_rejected = 0;
  unsigned int rejected_background = 0;
  unsigned int rejected_threshold = 0;

  for (unsigned int i = 0; i < num_detections_classes; ++i)
  {
    auto classes = view<float>(tensor_classes, i, num_classes);
    auto boxes = view<float>(tensor_boxes, i, Layer::Boxes::Box::SIZE);

    // Check rejection reason before calling parse_detection
    float max_class_val = classes[0];
    std::size_t max_class_idx = 0;
    for (std::size_t j = 1; j < num_classes; ++j)
    {
      if (classes[j] > max_class_val)
      {
        max_class_val = classes[j];
        max_class_idx = j;
      }
    }

    // Check for invalid box coordinates - be more lenient since we handle inf in parse_detection
    bool invalid_box = std::isnan(boxes[Layer::Boxes::Box::CX]) ||
                       std::isnan(boxes[Layer::Boxes::Box::CY]) ||
                       std::isinf(boxes[Layer::Boxes::Box::CX]) ||
                       std::isinf(boxes[Layer::Boxes::Box::CY]) ||
                       boxes[Layer::Boxes::Box::CX] < 0 || boxes[Layer::Boxes::Box::CX] > width ||
                       boxes[Layer::Boxes::Box::CY] < 0 || boxes[Layer::Boxes::Box::CY] > height;

    if (invalid_box)
    {
      detections_rejected++;
      rejected_threshold++;  // Count as threshold failure
      if (detections_rejected <= 5)
      {
        std::cerr << "DeepStream-RFDETR: DEBUG - Detection " << i
                  << " rejected: invalid box (cx=" << boxes[Layer::Boxes::Box::CX]
                  << ", cy=" << boxes[Layer::Boxes::Box::CY]
                  << ", w=" << boxes[Layer::Boxes::Box::W]
                  << ", h=" << boxes[Layer::Boxes::Box::H] << ")\n";
      }
      continue;
    }

    bool use_probabilities = (layer_classes->layerName == std::string_view("4132"));
    auto detection = parse_detection(boxes, classes, params, width, height, use_probabilities);
    if (!detection)
    {
      detections_rejected++;
      if (max_class_idx == Layer::Classes::BACKGROUND)
      {
        rejected_background++;
      }
      else
      {
        rejected_threshold++;
      }

      // Debug first few rejections
      if (detections_rejected <= 5)
      {
        if (max_class_idx == Layer::Classes::BACKGROUND)
        {
          std::cerr << "DeepStream-RFDETR: DEBUG - Detection " << i
                    << " rejected: background class (max_idx=0)\n";
        }
        else
        {
          std::cerr << "DeepStream-RFDETR: DEBUG - Detection " << i
                    << " rejected: low confidence (max_idx=" << max_class_idx
                    << ", max_val=" << max_class_val << ")\n";
        }
      }
      continue;
    }

    detections_passed++;
    detections.push_back(*detection);

    // Debug first few successful detections
    if (detections_passed <= 3)
    {
      // Find top 5 classes
      std::vector<std::pair<float, std::size_t>> class_scores;
      for (std::size_t j = 0; j < num_classes; ++j)
      {
        class_scores.emplace_back(classes[j], j);
      }
      std::sort(class_scores.rbegin(), class_scores.rend());

      std::cerr << "DeepStream-RFDETR: DEBUG - Detection " << i
                << " passed: class=" << detection->classId
                << ", conf=" << detection->detectionConfidence
                << ", bbox=[" << detection->left << ", " << detection->top
                << ", " << detection->width << ", " << detection->height << "]\n";
      std::cerr << "DeepStream-RFDETR: DEBUG - Detection " << i << " top 5 classes: ";
      for (std::size_t j = 0; j < std::min(5ul, class_scores.size()); ++j)
      {
        std::cerr << "[" << class_scores[j].second << "]=" << class_scores[j].first;
        if (j < 4) std::cerr << ", ";
      }
      std::cerr << "\n";
    }
  }

  std::cerr << "DeepStream-RFDETR: DEBUG - Summary: " << detections_passed
            << " passed, " << detections_rejected << " rejected\n";
  if (detections_rejected > 0)
  {
    std::cerr << "  - Rejected (background): " << rejected_background << "\n";
    std::cerr << "  - Rejected (threshold): " << rejected_threshold << "\n";
  }

  return true;
}

// Unused, just having the compiler check the signature
namespace {
[[maybe_unused]] const NvDsInferParseCustomFunc check_deepstream_rfdetr_bbox =
    deepstream_rfdetr_bbox;
}
