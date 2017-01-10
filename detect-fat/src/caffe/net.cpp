#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

//#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
//#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/upgrade_proto.hpp"

//#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/fast_rcnn_layers.hpp" 


#ifdef ANDROID_LOG
#include <android/log.h>
#endif

#define CHECK 1

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  Init(param);
}


template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages,
    const Net* root_net)
    : root_net_(root_net) {

  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

// ------ add by chigo, 2016-12-26 ------
template <typename Dtype>
Net<Dtype>::Net(const string& model_file, const int model_type,
	const int level, const vector<string>* stages, 
	const Net* root_net)
    : root_net_(root_net) {

	//printf("Net:21\n");
    if ( model_type == 1 )	//uint8
		LoadTrainedLayers_UINT8(model_file,caffe::TEST,level,stages);
	else if ( model_type == 0 )	//Dtype
		LoadTrainedLayers(model_file,caffe::TEST,level,stages);
	//printf("Net:22\n");
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  //CHECK(Caffe::root_solver() || root_net_)
  //    << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  //LOG_IF(INFO, Caffe::root_solver())
  //    << "Initializing net from parameters: " << std::endl
  //    << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  //CHECK(param.input_dim_size() == 0 || param.input_shape_size() == 0)
  //    << "Must specify either input_shape OR deprecated input_dim, not both.";
  if (param.input_dim_size() > 0) {
    // Deprecated 4D dimensions.
    //CHECK_EQ(param.input_size() * 4, param.input_dim_size())
    //    << "Incorrect input blob dimension specifications.";
  } else {
    //CHECK_EQ(param.input_size(), param.input_shape_size())
    //    << "Exactly one input_shape must be specified per input.";
  }
  memory_used_ = 0;
  // set the input blobs
  for (int input_id = 0; input_id < param.input_size(); ++input_id) {
    const int layer_id = -1;  // inputs have fake layer ID -1
    AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
  }
   
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    //std::cout<<layer_id<<std::endl;
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    //std::cout <<layer_id << " - Creating Layer " << layer_param.name() <<std::endl;;
    if (layer_param.propagate_down_size() > 0) {
      //CHECK_EQ(layer_param.propagate_down_size(),
      //    layer_param.bottom_size())
      //    << "propagate_down param must be specified "
      //    << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
        std::cout << "Sharing layer " << layer_param.name() << " from root net"<<std::endl;
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
      
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name());
    //LOG_IF(INFO, Caffe::root_solver())
    //std::cout    << "Creating Layer " << layer_param.name() <<std::endl;;
    bool need_backward = false;
    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        std::cout << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      //std::cout<<"--"<<layer_id<<" "<<bottom_vecs_.size()<<" "<<top_vecs_.size()<<std::endl;
       // std::cout<<layer_id<<"->"<<layers_[layer_id]->type()<<std::endl;
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    //LOG_IF(INFO, Caffe::root_solver())
    //std::cout    << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      //LOG_IF(INFO, Caffe::root_solver())
      //std::cout    << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        //LOG_IF(INFO, Caffe::root_solver())
        std::cout    << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    //LOG_IF(INFO, Caffe::root_solver())
    //    << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    //CHECK_LE(param_size, num_param_blobs)
    //    << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        //LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        //LOG(INFO) << layer_names_[layer_id]
        //    << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  /*
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  */
  
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    //LOG_IF(INFO, Caffe::root_solver())
    //    << "This network produces output " << *it;
    //std::cout<< "This network produces output " << *it<<std::endl;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  //LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
  //std::cout<<"Network initialization done:"<<"output_blobs[0]->count()="<<net_output_blobs_[0]->count()<<std::endl;

}

    
    
template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    //CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
    //      << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        //LOG_IF(INFO, Caffe::root_solver())
        //    << "The NetState phase (" << state.phase()
        //    << ") differed from the phase (" << rule.phase()
        //    << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      //LOG_IF(INFO, Caffe::root_solver())
      //    << "The NetState level (" << state.level()
      //    << ") is above the min_level (" << rule.min_level()
      //    << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      //LOG_IF(INFO, Caffe::root_solver())
      //    << "The NetState level (" << state.level()
      //    << ") is above the max_level (" << rule.max_level()
      //    << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      //LOG_IF(INFO, Caffe::root_solver())
      //    << "The NetState did not contain stage '" << rule.stage(i)
      //    << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      //LOG_IF(INFO, Caffe::root_solver())
      //    << "The NetState contained a not_stage '" << rule.not_stage(i)
      //    << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new input or top blob to the net.  (Inputs have
// layer_id == -1, tops have layer_id >= 0.)
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param((layer_id >= 0) ?
    (new LayerParameter(param.layer(layer_id))) : NULL);
  const string& blob_name = layer_param ?
      (layer_param->top_size() > top_id ?
          layer_param->top(top_id) : "(automatic)") : param.input(top_id);
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    //LOG_IF(INFO, Caffe::root_solver())
    //    << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    //LOG(FATAL) << "Top blob '" << blob_name
    //           << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      if (layer_param) {
        //LOG(INFO) << layer_param->name() << " -> " << blob_name;
      } else {
        //LOG(INFO) << "Input " << top_id << " -> " << blob_name;
      }
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    if (layer_id == -1) {
      // Set the (explicitly specified) dimensions of the input blob.
      if (param.input_dim_size() > 0) {
        blob_pointer->Reshape(param.input_dim(top_id * 4),
                              param.input_dim(top_id * 4 + 1),
                              param.input_dim(top_id * 4 + 2),
                              param.input_dim(top_id * 4 + 3));
      } else {
        blob_pointer->Reshape(param.input_shape(top_id));
      }
      net_input_blob_indices_.push_back(blob_id);
      net_input_blobs_.push_back(blob_pointer.get());
    } else {
      top_id_vecs_[layer_id].push_back(blob_id);
      top_vecs_[layer_id].push_back(blob_pointer.get());
    }
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    //LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
    //           << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  //LOG_IF(INFO, Caffe::root_solver())
  //    << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool propagate_down = true;
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0)
    propagate_down = layer_param.propagate_down(bottom_id);
  const bool need_backward = blob_need_backward_[blob_id] &&
                          propagate_down;
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    //LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
    //    << "' owned by "
    //    << "layer '" << layer_names_[owner_layer_id] << "', param "
    //    << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      //CHECK_EQ(this_blob->count(), owner_blob->count())
      //    << "Cannot share param '" << param_name << "' owned by layer '"
      //    << layer_names_[owner_layer_id] << "' with layer '"
      //    << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
      //    << "shape is " << owner_blob->shape_string() << "; sharing layer "
      //    << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      //CHECK(this_blob->shape() == owner_blob->shape())
      //    << "Cannot share param '" << param_name << "' owned by layer '"
      //    << layer_names_[owner_layer_id] << "' with layer '"
      //    << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
      //    << "shape is " << owner_blob->shape_string() << "; sharing layer "
      //    << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        //CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
        //    << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        //CHECK_EQ(param_spec->decay_mult(),
        //         params_weight_decay_[learnable_param_id])
        //    << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  //CHECK_GE(start, 0);
  //CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  //if (debug_info_) 
  {
    //for (int i = 0; i < net_input_blobs_.size(); ++i) {
    //  InputDebugInfo(i);
    //}
  }
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    //std::cout<<i<<"-th forward:"<<layers_[i]->type()<<" - "<<layer_names_[i].c_str()<<std::endl;
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    //if (debug_info_)  
    // ForwardDebugInfo(i); 
    
  }

  //std::cout<<"ForwardFromTo end."<<std::endl;
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled(Dtype* loss) {

  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return ForwardPrefilled(loss);
}

/*
template <typename Dtype>
string Net<Dtype>::Forward(const string& input_blob_protos, Dtype* loss) {
  //std::cout<<"net-Forward-630."<<std::endl;
  BlobProtoVector blob_proto_vec;
  if (net_input_blobs_.size()) {
    blob_proto_vec.ParseFromString(input_blob_protos);
    //CHECK_EQ(blob_proto_vec.blobs_size(), net_input_blobs_.size())
    //    << "Incorrect input size.";
    for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
      net_input_blobs_[i]->FromProto(blob_proto_vec.blobs(i));
    }
  }
  ForwardPrefilled(loss);
  blob_proto_vec.Clear();
  for (int i = 0; i < net_output_blobs_.size(); ++i) {
    net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs());
  }
  string output;
  blob_proto_vec.SerializeToString(&output);
  return output;
}
*/

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    //CopyTrainedLayersFromHDF5(trained_filename);
    std::cout<<"CopyTrainedLayersFromHDF5."<<std::endl;
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }

    if (target_layer_id == layer_names_.size()) {
      //LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    //DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    //CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
    //    << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        //LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
        //    << source_layer_name << "'; shape mismatch.  Source param shape is "
        //    << source_blob.shape_string() << "; target param shape is "
        //    << target_blobs[j]->shape_string() << ". "
        //    << "To learn this layer's parameters from scratch rather than "
        //    << "copying from a saved net, rename the layer.";

      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}



template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      //LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    //DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    //CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
    //    << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      //CHECK(target_blobs[j]->shape() == source_blob->shape())
      //    << "Cannot share param " << j << " weights from layer '"
      //    << source_layer_name << "'; shape mismatch.  Source param shape is "
      //    << source_blob->shape_string() << "; target param shape is "
      //    << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}


template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}


 // ------ add by tangyuan, 2016-11-26 ------
template <typename Dtype>
  int Net<Dtype>::LoadTrainedLayers(const std::string& compress_file, Phase phase, 
  		const int level, const vector<string>* stages)
  {
    //
    //std::string compress_file("save.dat");
    std::ifstream is(compress_file.c_str(), std::ios::binary);
	
    //
    //std::cout<<"start read_string..."<<std::endl;
    std::string param_string;
    read_string(is, param_string);
    //std::cout<<param_string<<std::endl;

	//std::cout<<"start ReadNetParamsFromTextStringOrDie..."<<std::endl;
    NetParameter param;
    ReadNetParamsFromTextStringOrDie(param_string, &param);

	// Set stages and level
	//std::cout<<"start Init..."<<std::endl;
	param.mutable_state()->set_phase(phase);
	if (stages != NULL) {
	  for (int i = 0; i < stages->size(); i++) {
	    param.mutable_state()->add_stage((*stages)[i]);
	  }
	}
	param.mutable_state()->set_level(level);
    Init(param);

    //
    //std::cout<<"read layer..."<<std::endl;
    shared_ptr<Layer<Dtype> > layer;
    std::vector<shared_ptr<Blob<Dtype> > > blobs;
    for (unsigned int i = 0; i < layers_.size(); ++i) {
       layer = layers_[i];
       blobs = layer->blobs();
       //std::cout<<i<<"-th layer:"<<layer->type()<<" blob size:"<<blobs.size()<<std::endl;

       //for(int b = 0; b < blobs.size(); b++)
       //   std::cout<<"    "<<blobs[b]->width()<<"x"<<blobs[b]->height()
       //       <<"x"<<blobs[b]->channels()<<"x"<<blobs[b]->num()<<std::endl;
        
        for(int b = 0; b < blobs.size(); b++){
          read_real<Dtype>(is, blobs[b]->mutable_cpu_data(), blobs[b]->count());
        }
    }//for-i
    is.close();
    return 0;
  }

  // ------ add by tangyuan, 2016-11-26 ------
template <typename Dtype>
  int Net<Dtype>::SaveTrainedLayers(const std::string& deploy_file, const std::string& compress_file)
  {
    //
    std::ifstream ifs(deploy_file);
    // assign
    std::string param_string( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );
    ifs.close();

    //std::string compress_file("save.dat");
    std::ofstream os(compress_file.c_str(), std::ios::binary);
    write_string(os, param_string);
    //
    shared_ptr<Layer<Dtype> > layer;
    std::vector<shared_ptr<Blob<Dtype> > > blobs;
    for (unsigned int i = 0; i < layers_.size(); ++i) {
       layer = layers_[i];
       blobs = layer->blobs();
       //std::cout<<i<<"-th layer:"<<layer->type()<<" blob size:"<<blobs.size()<<std::endl;

       //for(int b = 0; b < blobs.size(); b++)
       //   std::cout<<"    "<<blobs[b]->width()<<"x"<<blobs[b]->height()
       //       <<"x"<<blobs[b]->channels()<<"x"<<blobs[b]->num()<<std::endl;
        
        for(int b = 0; b<blobs.size(); b++){
          write_real<Dtype>(os, blobs[b]->mutable_cpu_data(), blobs[b]->count());
        }
    }//for-i
    os.close();
    return 0;
  }

// ------ add by chigo, 2016-12-26 ------
template <typename Dtype>
	int Net<Dtype>::LoadTrainedLayers_UINT8(const std::string& compress_file, Phase phase, 
		  const int level, const vector<string>* stages)
{
	Dtype tmp;
	unsigned int i,j,k;
	unsigned long long len = 0;
	
	//
	//std::string compress_file("save.dat");
	std::ifstream is(compress_file.c_str(), std::ios::binary);

	//
	//std::cout<<"start read_string..."<<std::endl;
	std::string param_string;
	read_string(is, param_string);
	//std::cout<<param_string<<std::endl;

	//std::cout<<"start ReadNetParamsFromTextStringOrDie..."<<std::endl;
	NetParameter param;
	ReadNetParamsFromTextStringOrDie(param_string, &param);

	// Set stages and level
	//std::cout<<"start Init..."<<std::endl;
	param.mutable_state()->set_phase(phase);
	if (stages != NULL) {
		for (int i = 0; i < stages->size(); i++) {
		  param.mutable_state()->add_stage((*stages)[i]);
		}
	}
	param.mutable_state()->set_level(level);
	Init(param);

	//
	//std::cout<<"read layer..."<<std::endl;
	shared_ptr<Layer<Dtype> > layer;
	std::vector<shared_ptr<Blob<Dtype> > > blobs;
	for (i = 0; i < layers_.size(); ++i) {
		layer = layers_[i];
		blobs = layer->blobs();
		//std::cout<<i<<"-th layer:"<<layer->type()<<" blob size:"<<blobs.size()<<std::endl;

		for(j = 0; j < blobs.size(); j++){
			//read_real<Dtype>(is, blobs[j]->mutable_cpu_data(), blobs[j]->count());

			len = blobs[j]->count();
			Dtype* blob_data = blobs[j]->mutable_cpu_data();
			Dtype* minmax = new Dtype[2];//0-min,1-max
			unsigned char* ucharBolbData = new unsigned char[len];
			memset(ucharBolbData,0,len);

			//read data
			read_real<Dtype>(is, minmax, 2);
			read_real<unsigned char>(is, ucharBolbData, len);

			//Normalization
			tmp = (minmax[1] - minmax[0])/255.0;
			for(k = 0; k < len; k++)
			{
				blob_data[k] = (Dtype)( ucharBolbData[k]*tmp + minmax[0] ) ;
			}

			//check
			//std::cout<<"    "<<blobs[j]->width()<<"x"<<blobs[j]->height()
		    //  <<"x"<<blobs[j]->channels()<<"x"<<blobs[j]->num()
		    //  <<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;
			
			if (minmax) {delete [] minmax;minmax = 0;}
			if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
		}
	}//for-i
	is.close();
	return 0;
}

  // ------ add by chigo, 2016-12-26 ------
template <typename Dtype>
  int Net<Dtype>::SaveTrainedLayers_UINT8(const std::string& deploy_file, const std::string& compress_file)
  {
  	Dtype tmp;
  	unsigned int i,j,k;
	unsigned long long len = 0;
	
    //
    std::ifstream ifs(deploy_file);
    // assign
    std::string param_string( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );
    ifs.close();

    //std::string compress_file("save.dat");
    std::ofstream os(compress_file.c_str(), std::ios::binary);
    write_string(os, param_string);
    //
    shared_ptr<Layer<Dtype> > layer;
    std::vector<shared_ptr<Blob<Dtype> > > blobs;
    for (i = 0; i < layers_.size(); ++i) {
		layer = layers_[i];
		blobs = layer->blobs();
		//std::cout<<i<<"-th layer:"<<layer->type()<<" blob size:"<<blobs.size()<<std::endl;
        
		for(j = 0; j < blobs.size(); j++){
			//write_real<Dtype>(os, blobs[b]->mutable_cpu_data(), blobs[b]->count());

			len = blobs[j]->count();
			Dtype* blob_data = blobs[j]->mutable_cpu_data();
			Dtype* minmax = new Dtype[2];//0-min,1-max
			unsigned char* ucharBolbData = new unsigned char[len];
			memset(ucharBolbData,0,len);
			
			//get minmax
			minmax[0] = 1000000.0;
			minmax[1] = -1000000.0;
			for(k = 0; k < len; k++)
			{
				if ( blob_data[k] < minmax[0] )
					minmax[0] = blob_data[k];
				if ( blob_data[k] > minmax[1] )
					minmax[1] = blob_data[k];
			}

			//Normalization
			if ( minmax[0] != minmax[1] )
			{
				tmp = 255.0/(minmax[1] - minmax[0]);
				for(k = 0; k < len; k++)
				{
					ucharBolbData[k] = (unsigned char)((blob_data[k]-minmax[0])*tmp + 0.5) ;
				}
			}

			//write data
			write_real<Dtype>(os, minmax, 2);
			write_real<unsigned char>(os, ucharBolbData, len);

			//check
			//std::cout<<"    "<<blobs[j]->width()<<"x"<<blobs[j]->height()
		    //  <<"x"<<blobs[j]->channels()<<"x"<<blobs[j]->num()
		    //  <<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;

			if (minmax) {delete [] minmax;minmax = 0;}
			if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
      	}
    }//for-i
    os.close();
    return 0;
  }

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    //LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    //LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

//
template <typename Dtype>
 void  Net<Dtype>::SaveNet(const std::string& m_file)
{
    // shared_ptr<Net<Dtype> > other
    std::cout<<"save net..."<<std::endl;
    //
    std::ofstream os(m_file.c_str(), std::ios::binary);
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    std::cout<<" save blobs."<<std::endl;
    write_real<size_t>(os, this->blobs_.size());
   for(int b = 0; b<blobs_.size(); b++){
        //
        write_string(os, blob_names_[b]);
        write_real<int>(os, blobs_[b]->num());
        write_real<int>(os, blobs_[b]->channels());
        write_real<int>(os, blobs_[b]->width());
        write_real<int>(os, blobs_[b]->height());
    }
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    std::cout<<" save layers."<<std::endl;
    shared_ptr<Layer<Dtype> > layer;
    write_real<size_t>(os, layers_.size());
    for (unsigned int i = 0; i < layers_.size(); ++i) {
        //
        layer = layers_[i];
        //
        write_string(os, layer->type());
        write_string(os, layer_names_[i]);
        
        //
        write_real<size_t>(os, bottom_id_vecs_[i].size());
        for(int b = 0; b<bottom_id_vecs_[i].size(); b++){
            write_real<int>(os, bottom_id_vecs_[i][b]);
          }
        write_real<size_t>(os, top_id_vecs_[i].size());
        for(int t = 0; t<top_id_vecs_[i].size(); t++){
            write_real<int>(os, this->top_id_vecs()[i][t]);
          }
        //
        if( layer->type() == "Split"){

        }
        else if( layer->type() == "ReLU"){

            ReLULayer<Dtype>* p = dynamic_cast<ReLULayer<Dtype>*>(layer.get());
            write_real<Dtype>(os, p->negative_slope() );

			//const ReLUParameter& relu_param = this->layer_param_.relu_param();
  			//Dtype negative_slope = relu_param.negative_slope();
        }
        else if( layer->type() == "Concat"){
           
            ConcatLayer<Dtype>* p = dynamic_cast<ConcatLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->concat_axis());

			//const ConcatParameter& concat_param = this->layer_param_.concat_param();
        }
        else if( layer->type() == "Dropout"){

        }
        else if( layer->type() == "Pooling"){

            //
            PoolingLayer<Dtype>* p = dynamic_cast<PoolingLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->global_pooling());
            write_real<int>(os, (int)p->layer_param().pooling_param().pool());
            write_real<int>(os, p->kernel_h());
            write_real<int>(os, p->kernel_w());
            write_real<int>(os, p->pad_h());
            write_real<int>(os, p->pad_w());
            write_real<int>(os, p->stride_h());
            write_real<int>(os, p->stride_w());

			//const PoolingParameter& pool_param = this->layer_param_.pooling_param();

        }
        else if( layer->type() == "Softmax"){

            //
            SoftmaxLayer<Dtype>* p = dynamic_cast<SoftmaxLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->layer_param().softmax_param().axis());

			//const SoftmaxParameter& softmax_param = this->layer_param_.softmax_param();
        }
        else if( layer->type() == "Convolution"){

            //
            ConvolutionLayer<Dtype>* p = dynamic_cast<ConvolutionLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
            write_real<size_t>(os, layer->blobs().size());
            for(int b = 0; b < layer->blobs().size(); b++){
                write_real<int>(os, layer->blobs()[b]->num());
                write_real<int>(os, layer->blobs()[b]->channels());
                write_real<int>(os, layer->blobs()[b]->width());
                write_real<int>(os, layer->blobs()[b]->height());
                write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
             }//for
            }//if
        
        }//if
    }//for
    
    
    
    //
    os.close();

}

//
template <typename Dtype>
 void  Net<Dtype>::SaveNet_int8(const std::string& m_file)
{
    // shared_ptr<Net<Dtype> > other
    std::cout<<"save net..."<<std::endl;
    //
    std::ofstream os(m_file.c_str(), std::ios::binary);
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    std::cout<<" save blobs."<<std::endl;
    write_real<size_t>(os, this->blobs_.size());
   for(int b = 0; b<blobs_.size(); b++){
        //
        write_string(os, blob_names_[b]);
        write_real<int>(os, blobs_[b]->num());
        write_real<int>(os, blobs_[b]->channels());
        write_real<int>(os, blobs_[b]->width());
        write_real<int>(os, blobs_[b]->height());
    }
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    std::cout<<" save layers."<<std::endl;
    shared_ptr<Layer<Dtype> > layer;
    write_real<size_t>(os, layers_.size());
    for (unsigned int i = 0; i < layers_.size(); ++i) {
        //
        layer = layers_[i];
        //
        write_string(os, layer->type());
        write_string(os, layer_names_[i]);
        
        //
        write_real<size_t>(os, bottom_id_vecs_[i].size());
        for(int b = 0; b<bottom_id_vecs_[i].size(); b++){
            write_real<int>(os, bottom_id_vecs_[i][b]);
          }
        write_real<size_t>(os, top_id_vecs_[i].size());
        for(int t = 0; t<top_id_vecs_[i].size(); t++){
            write_real<int>(os, this->top_id_vecs()[i][t]);
          }
        //
        if( layer->type() == "Split"){

        }
        else if( layer->type() == "ReLU"){

            ReLULayer<Dtype>* p = dynamic_cast<ReLULayer<Dtype>*>(layer.get());
            write_real<Dtype>(os, p->negative_slope() );

			//const ReLUParameter& relu_param = this->layer_param_.relu_param();
  			//Dtype negative_slope = relu_param.negative_slope();
        }
        else if( layer->type() == "Concat"){
           
            ConcatLayer<Dtype>* p = dynamic_cast<ConcatLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->concat_axis());

			//const ConcatParameter& concat_param = this->layer_param_.concat_param();
        }
        else if( layer->type() == "Dropout"){

        }
        else if( layer->type() == "Pooling"){

            //
            PoolingLayer<Dtype>* p = dynamic_cast<PoolingLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->global_pooling());
            write_real<int>(os, (int)p->layer_param().pooling_param().pool());
            write_real<int>(os, p->kernel_h());
            write_real<int>(os, p->kernel_w());
            write_real<int>(os, p->pad_h());
            write_real<int>(os, p->pad_w());
            write_real<int>(os, p->stride_h());
            write_real<int>(os, p->stride_w());

			//const PoolingParameter& pool_param = this->layer_param_.pooling_param();

        }
        else if( layer->type() == "Softmax"){

            //
            SoftmaxLayer<Dtype>* p = dynamic_cast<SoftmaxLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->layer_param().softmax_param().axis());

			//const SoftmaxParameter& softmax_param = this->layer_param_.softmax_param();
        }
        else if( layer->type() == "Convolution"){

            //
            ConvolutionLayer<Dtype>* p = dynamic_cast<ConvolutionLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
				//write_real<size_t>(os, layer->blobs().size());
	            //for(int b = 0; b < layer->blobs().size(); b++){
	            //    write_real<int>(os, layer->blobs()[b]->num());
	            //    write_real<int>(os, layer->blobs()[b]->channels());
	            //    write_real<int>(os, layer->blobs()[b]->width());
	            //    write_real<int>(os, layer->blobs()[b]->height());
	            //    write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
	            // }//for

				//data change
				blob_float2uchar(os, layer->blobs());
            }//if
        
        }//if
    }//for
    
    //
    os.close();

}

//
template <typename Dtype>
 void  Net<Dtype>::SaveNet_detect(const std::string& m_file)
{
    // shared_ptr<Net<Dtype> > other
    std::cout<<"save net..."<<std::endl;
    //
    std::ofstream os(m_file.c_str(), std::ios::binary);
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    std::cout<<" save blobs."<<std::endl;
    write_real<size_t>(os, this->blobs_.size());
   for(int b = 0; b<blobs_.size(); b++){
        //
        write_string(os, blob_names_[b]);
        write_real<int>(os, blobs_[b]->num());
        write_real<int>(os, blobs_[b]->channels());
        write_real<int>(os, blobs_[b]->width());
        write_real<int>(os, blobs_[b]->height());
    }
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    std::cout<<" save layers."<<std::endl;
    shared_ptr<Layer<Dtype> > layer;
    write_real<size_t>(os, layers_.size());
    for (unsigned int i = 0; i < layers_.size(); ++i) {
        //
        layer = layers_[i];
        //
        write_string(os, layer->type());
        write_string(os, layer_names_[i]);
        
        //
        write_real<size_t>(os, bottom_id_vecs_[i].size());
        for(int b = 0; b<bottom_id_vecs_[i].size(); b++){
            write_real<int>(os, bottom_id_vecs_[i][b]);
          }
        write_real<size_t>(os, top_id_vecs_[i].size());
        for(int t = 0; t<top_id_vecs_[i].size(); t++){
            write_real<int>(os, this->top_id_vecs()[i][t]);
          }
        //
        if( layer->type() == "Split"){
			printf( "type:%s\n", layer->type() );

        }
        else if( layer->type() == "ReLU"){
			printf( "type:%s\n", layer->type() );

            ReLULayer<Dtype>* p = dynamic_cast<ReLULayer<Dtype>*>(layer.get());
            write_real<Dtype>(os, p->negative_slope() );

			//const ReLUParameter& relu_param = this->layer_param_.relu_param();
  			//Dtype negative_slope = relu_param.negative_slope();
        }
        else if( layer->type() == "Concat"){
			printf( "type:%s\n", layer->type() );
           
            ConcatLayer<Dtype>* p = dynamic_cast<ConcatLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->concat_axis());

			//const ConcatParameter& concat_param = this->layer_param_.concat_param();
        }
        else if( layer->type() == "Dropout"){
			printf( "type:%s\n", layer->type() );

        }
        else if( layer->type() == "Pooling"){
			printf( "type:%s\n", layer->type() );

            //
            PoolingLayer<Dtype>* p = dynamic_cast<PoolingLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->global_pooling());
            write_real<int>(os, (int)p->layer_param().pooling_param().pool());
            write_real<int>(os, p->kernel_h());
            write_real<int>(os, p->kernel_w());
            write_real<int>(os, p->pad_h());
            write_real<int>(os, p->pad_w());
            write_real<int>(os, p->stride_h());
            write_real<int>(os, p->stride_w());

			//printf( "type:%s,pool:%d\n", layer->type(), (int)p->layer_param().pooling_param().pool() );
			//const PoolingParameter& pool_param = this->layer_param_.pooling_param();
        }
        else if( layer->type() == "Softmax"){
			printf( "type:%s\n", layer->type() );

            //
            SoftmaxLayer<Dtype>* p = dynamic_cast<SoftmaxLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->layer_param().softmax_param().axis());

			//const SoftmaxParameter& softmax_param = this->layer_param_.softmax_param();
        }
        else if( layer->type() == "Convolution"){
			printf( "type:%s\n", layer->type() );

            //
            ConvolutionLayer<Dtype>* p = dynamic_cast<ConvolutionLayer<Dtype>*>(layer.get());

            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
            write_real<size_t>(os, layer->blobs().size());
            for(int b = 0; b < layer->blobs().size(); b++){
                write_real<int>(os, layer->blobs()[b]->num());
                write_real<int>(os, layer->blobs()[b]->channels());
                write_real<int>(os, layer->blobs()[b]->width());
                write_real<int>(os, layer->blobs()[b]->height());
                write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
             }//for
            }//if
        
        }//if
        else if( layer->type() == "Reshape"){
			printf( "type:%s\n", layer->type() );
			
			ReshapeLayer<Dtype>* p = dynamic_cast<ReshapeLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->axis());
			write_real<int>(os, p->num_axes());
			
			printf( "type:%s,axis:%d,num_axes:%d,shape.size:%ld,", 
				layer->type(),p->axis(),p->num_axes(),p->shape().size() );
			if(p->shape().size() > 0)
			{
            	write_real<size_t>(os, p->shape().size());
            	for(int b = 0; b < p->shape().size(); b++){
                	write_real<int>(os, p->shape()[b]);
					printf( "%d_", p->shape()[b] );
            	}
            }
			printf( "\n");
        }
        else if( layer->type() == "Deconvolution"){
			printf( "type:%s\n", layer->type() );

            //
            DeconvolutionLayer<Dtype>* p = dynamic_cast<DeconvolutionLayer<Dtype>*>(layer.get());

            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
            write_real<size_t>(os, layer->blobs().size());
            for(int b = 0; b < layer->blobs().size(); b++){
                write_real<int>(os, layer->blobs()[b]->num());
                write_real<int>(os, layer->blobs()[b]->channels());
                write_real<int>(os, layer->blobs()[b]->width());
                write_real<int>(os, layer->blobs()[b]->height());
                write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
             }//for
            }//if
        }
		else if( layer->type() == "ProposalLayer"){
			printf( "type:%s\n", layer->type() );
			
			ProposalLayer<Dtype>* p = dynamic_cast<ProposalLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->base_size());
			write_real<int>(os, p->feat_stride());
			write_real<int>(os, p->pre_nms_topn());
			write_real<int>(os, p->post_nms_topn());
			write_real<Dtype>(os, p->nms_thresh());
			write_real<int>(os, p->min_size());
			
			printf( "type:%s,base_size:%d,feat_stride:%d,pre_nms_topn:%d,post_nms_topn:%d,nms_thresh:%.4f,min_size:%d,vecRatios.size:%ld,", 
				layer->type(),p->base_size(),p->feat_stride(),p->pre_nms_topn(),p->post_nms_topn(),
				p->nms_thresh(),p->min_size(),p->vecRatios().size() );
			if(p->vecRatios().size() > 0)
			{
            	write_real<size_t>(os, p->vecRatios().size());
            	for(int b = 0; b < p->vecRatios().size(); b++){
                	write_real<Dtype>(os, p->vecRatios()[b]);
					printf( "%.4f_", p->vecRatios()[b] );
            	}
            }
			
			printf( ",vecScales.size:%ld,",p->vecScales().size());
			if(p->vecScales().size() > 0)
			{
            	write_real<size_t>(os, p->vecScales().size());
            	for(int b = 0; b < p->vecScales().size(); b++){
                	write_real<Dtype>(os, p->vecScales()[b]);
					printf( "%.4f_", p->vecScales()[b] );
            	}
            }
			printf( "\n");
        }
		else if( layer->type() == "ROIPooling"){
			printf( "type:%s\n", layer->type() );

			ROIPoolingLayer<Dtype>* p = dynamic_cast<ROIPoolingLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->pooled_width());
			write_real<int>(os, p->pooled_height());
			write_real<Dtype>(os, p->spatial_scale());
			
			printf( "type:%s,pooled_width:%d,pooled_height:%d,spatial_scale:%.4f\n", 
				layer->type(),p->pooled_width(),p->pooled_height(),p->spatial_scale() );
        }
		else if( layer->type() == "InnerProduct"){
			printf( "type:%s\n", layer->type() );

			InnerProductLayer<Dtype>* p = dynamic_cast<InnerProductLayer<Dtype>*>(layer.get());

			write_real<int>(os, (int)p->bias_term());
			write_real<int>(os, (int)p->transpose());
			write_real<int>(os, p->num_output());
			write_real<int>(os, p->axis());

			printf( "type:%s,bias_term:%d,transpose:%d,num_output:%d,axis:%d\n", 
				layer->type(),(int)p->bias_term(),(int)p->transpose(),p->num_output(),p->axis() );

			//
            if(layer->blobs().size() > 0){
            write_real<size_t>(os, layer->blobs().size());
            for(int b = 0; b < layer->blobs().size(); b++){
                write_real<int>(os, layer->blobs()[b]->num());
                write_real<int>(os, layer->blobs()[b]->channels());
                write_real<int>(os, layer->blobs()[b]->width());
                write_real<int>(os, layer->blobs()[b]->height());
                write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
             }//for
            }//if
            
        }
    }//for
    
    
    
    //
    os.close();

}

//
template <typename Dtype>
 void  Net<Dtype>::SaveNet_detect_int8(const std::string& m_file)
{
    // shared_ptr<Net<Dtype> > other
    std::cout<<"save net..."<<std::endl;
    //
    std::ofstream os(m_file.c_str(), std::ios::binary);
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    std::cout<<" save blobs."<<std::endl;
    write_real<size_t>(os, this->blobs_.size());
   for(int b = 0; b<blobs_.size(); b++){
        //
        write_string(os, blob_names_[b]);
        write_real<int>(os, blobs_[b]->num());
        write_real<int>(os, blobs_[b]->channels());
        write_real<int>(os, blobs_[b]->width());
        write_real<int>(os, blobs_[b]->height());
    }
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    std::cout<<" save layers."<<std::endl;
    shared_ptr<Layer<Dtype> > layer;
    write_real<size_t>(os, layers_.size());
    for (unsigned int i = 0; i < layers_.size(); ++i) {
        //
        layer = layers_[i];
        //
        write_string(os, layer->type());
        write_string(os, layer_names_[i]);
        
        //
        write_real<size_t>(os, bottom_id_vecs_[i].size());
        for(int b = 0; b<bottom_id_vecs_[i].size(); b++){
            write_real<int>(os, bottom_id_vecs_[i][b]);
          }
        write_real<size_t>(os, top_id_vecs_[i].size());
        for(int t = 0; t<top_id_vecs_[i].size(); t++){
            write_real<int>(os, this->top_id_vecs()[i][t]);
          }
        //
        if( layer->type() == "Split"){
			printf( "type:%s\n", layer->type() );

        }
        else if( layer->type() == "ReLU"){
			printf( "type:%s\n", layer->type() );

            ReLULayer<Dtype>* p = dynamic_cast<ReLULayer<Dtype>*>(layer.get());
            write_real<Dtype>(os, p->negative_slope() );

			//const ReLUParameter& relu_param = this->layer_param_.relu_param();
  			//Dtype negative_slope = relu_param.negative_slope();
        }
        else if( layer->type() == "Concat"){
			printf( "type:%s\n", layer->type() );
           
            ConcatLayer<Dtype>* p = dynamic_cast<ConcatLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->concat_axis());

			//const ConcatParameter& concat_param = this->layer_param_.concat_param();
        }
        else if( layer->type() == "Dropout"){
			printf( "type:%s\n", layer->type() );

        }
        else if( layer->type() == "Pooling"){
			printf( "type:%s\n", layer->type() );

            //
            PoolingLayer<Dtype>* p = dynamic_cast<PoolingLayer<Dtype>*>(layer.get());
            write_real<int>(os, (int)p->global_pooling());
            write_real<int>(os, (int)p->layer_param().pooling_param().pool());
            write_real<int>(os, p->kernel_h());
            write_real<int>(os, p->kernel_w());
            write_real<int>(os, p->pad_h());
            write_real<int>(os, p->pad_w());
            write_real<int>(os, p->stride_h());
            write_real<int>(os, p->stride_w());

			//printf( "type:%s,pool:%d\n", layer->type(), (int)p->layer_param().pooling_param().pool() );
			//const PoolingParameter& pool_param = this->layer_param_.pooling_param();
        }
        else if( layer->type() == "Softmax"){
			printf( "type:%s\n", layer->type() );

            //
            SoftmaxLayer<Dtype>* p = dynamic_cast<SoftmaxLayer<Dtype>*>(layer.get());
            write_real<int>(os, p->layer_param().softmax_param().axis());

			//const SoftmaxParameter& softmax_param = this->layer_param_.softmax_param();
        }
        else if( layer->type() == "Convolution"){
			printf( "type:%s\n", layer->type() );

            //
            ConvolutionLayer<Dtype>* p = dynamic_cast<ConvolutionLayer<Dtype>*>(layer.get());

            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
	            //write_real<size_t>(os, layer->blobs().size());
	            //for(int b = 0; b < layer->blobs().size(); b++){
	            //    write_real<int>(os, layer->blobs()[b]->num());
	            //    write_real<int>(os, layer->blobs()[b]->channels());
	            //    write_real<int>(os, layer->blobs()[b]->width());
	            //    write_real<int>(os, layer->blobs()[b]->height());
	            //    write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
	            // }//for

				//data change
				blob_float2uchar(os, layer->blobs());
            }//if
        
        }//if
        else if( layer->type() == "Reshape"){
			printf( "type:%s\n", layer->type() );
			
			ReshapeLayer<Dtype>* p = dynamic_cast<ReshapeLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->axis());
			write_real<int>(os, p->num_axes());
			
			printf( "type:%s,axis:%d,num_axes:%d,shape.size:%ld,", 
				layer->type(),p->axis(),p->num_axes(),p->shape().size() );
			if(p->shape().size() > 0)
			{
            	write_real<size_t>(os, p->shape().size());
            	for(int b = 0; b < p->shape().size(); b++){
                	write_real<int>(os, p->shape()[b]);
					printf( "%d_", p->shape()[b] );
            	}
            }
			printf( "\n");
        }
        else if( layer->type() == "Deconvolution"){
			printf( "type:%s\n", layer->type() );

            //
            DeconvolutionLayer<Dtype>* p = dynamic_cast<DeconvolutionLayer<Dtype>*>(layer.get());

            write_real<int>(os, (int)p->force_nd_im2col());
            write_real<int>(os, p->channel_axis());
            write_real<int>(os, p->num_spatial_axes());
            //
            write_real<int>(os, p->kernel_shape().num());
            write_real<int>(os, p->kernel_shape().channels());
            write_real<int>(os, p->kernel_shape().width());
            write_real<int>(os, p->kernel_shape().height());
            write_real<int>(os, p->kernel_shape().cpu_data(), p->kernel_shape().count());
            write_real<int>(os, p->stride().num());
            write_real<int>(os, p->stride().channels());
            write_real<int>(os, p->stride().width());
            write_real<int>(os, p->stride().height());
            write_real<int>(os, p->stride().cpu_data(), p->stride().count());
            write_real<int>(os, p->pad().num());
            write_real<int>(os, p->pad().channels());
            write_real<int>(os, p->pad().width());
            write_real<int>(os, p->pad().height());
            write_real<int>(os, p->pad().cpu_data(), p->pad().count());
            write_real<int>(os, p->dilation().num());
            write_real<int>(os, p->dilation().channels());
            write_real<int>(os, p->dilation().width());
            write_real<int>(os, p->dilation().height());
            write_real<int>(os, p->dilation().cpu_data(), p->dilation().count());
            //
            write_real<int>(os, (int)p->is_1x1());
            write_real<int>(os, p->channels());
            write_real<int>(os, p->num_output());
            write_real<int>(os, p->group());
            write_real<int>(os, p->conv_out_channels());
            write_real<int>(os, p->conv_in_channels());
            write_real<int>(os, (int)p->bias_term());
            write_real<int>(os, p->kernel_dim()); 
            //
            if(layer->blobs().size() > 0){
				//write_real<size_t>(os, layer->blobs().size());
	            //for(int b = 0; b < layer->blobs().size(); b++){
	            //    write_real<int>(os, layer->blobs()[b]->num());
	            //    write_real<int>(os, layer->blobs()[b]->channels());
	            //    write_real<int>(os, layer->blobs()[b]->width());
	            //    write_real<int>(os, layer->blobs()[b]->height());
	            //    write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
	            // }//for

				//data change
				blob_float2uchar(os, layer->blobs());
            }//if
        }
		else if( layer->type() == "ProposalLayer"){
			printf( "type:%s\n", layer->type() );
			
			ProposalLayer<Dtype>* p = dynamic_cast<ProposalLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->base_size());
			write_real<int>(os, p->feat_stride());
			write_real<int>(os, p->pre_nms_topn());
			write_real<int>(os, p->post_nms_topn());
			write_real<Dtype>(os, p->nms_thresh());
			write_real<int>(os, p->min_size());
			
			printf( "type:%s,base_size:%d,feat_stride:%d,pre_nms_topn:%d,post_nms_topn:%d,nms_thresh:%.4f,min_size:%d,vecRatios.size:%ld,", 
				layer->type(),p->base_size(),p->feat_stride(),p->pre_nms_topn(),p->post_nms_topn(),
				p->nms_thresh(),p->min_size(),p->vecRatios().size() );
			if(p->vecRatios().size() > 0)
			{
            	write_real<size_t>(os, p->vecRatios().size());
            	for(int b = 0; b < p->vecRatios().size(); b++){
                	write_real<Dtype>(os, p->vecRatios()[b]);
					printf( "%.4f_", p->vecRatios()[b] );
            	}
            }
			
			printf( ",vecScales.size:%ld,",p->vecScales().size());
			if(p->vecScales().size() > 0)
			{
            	write_real<size_t>(os, p->vecScales().size());
            	for(int b = 0; b < p->vecScales().size(); b++){
                	write_real<Dtype>(os, p->vecScales()[b]);
					printf( "%.4f_", p->vecScales()[b] );
            	}
            }
			printf( "\n");
        }
		else if( layer->type() == "ROIPooling"){
			printf( "type:%s\n", layer->type() );

			ROIPoolingLayer<Dtype>* p = dynamic_cast<ROIPoolingLayer<Dtype>*>(layer.get());
			write_real<int>(os, p->pooled_width());
			write_real<int>(os, p->pooled_height());
			write_real<Dtype>(os, p->spatial_scale());
			
			printf( "type:%s,pooled_width:%d,pooled_height:%d,spatial_scale:%.4f\n", 
				layer->type(),p->pooled_width(),p->pooled_height(),p->spatial_scale() );
        }
		else if( layer->type() == "InnerProduct"){
			printf( "type:%s\n", layer->type() );

			InnerProductLayer<Dtype>* p = dynamic_cast<InnerProductLayer<Dtype>*>(layer.get());

			write_real<int>(os, (int)p->bias_term());
			write_real<int>(os, (int)p->transpose());
			write_real<int>(os, p->num_output());
			write_real<int>(os, p->axis());

			printf( "type:%s,bias_term:%d,transpose:%d,num_output:%d,axis:%d\n", 
				layer->type(),(int)p->bias_term(),(int)p->transpose(),p->num_output(),p->axis() );

			//
            if(layer->blobs().size() > 0){
	            //write_real<size_t>(os, layer->blobs().size());
	            //for(int b = 0; b < layer->blobs().size(); b++){
	            //    write_real<int>(os, layer->blobs()[b]->num());
	            //    write_real<int>(os, layer->blobs()[b]->channels());
	            //    write_real<int>(os, layer->blobs()[b]->width());
	            //    write_real<int>(os, layer->blobs()[b]->height());
	            //    write_real<Dtype>(os, layer->blobs()[b]->cpu_data(), layer->blobs()[b]->count());
	            // }//for

				//data change
				blob_float2uchar(os, layer->blobs());
            }//if
            
        }
    }//for
    
    //
    os.close();

}

template <typename Dtype>
 void  Net<Dtype>::blob_float2uchar(std::ostream &os, vector<shared_ptr<Blob<Dtype>>> &inBlobs)
{
	//
	int j,k;
	size_t len,blob_size;
	Dtype tmp;

	//blob.size
	blob_size = inBlobs.size();
	write_real<size_t>(os, blob_size);
	    
	for(j = 0; j < blob_size; j++){
		//write_real<Dtype>(os, blobs[b]->mutable_cpu_data(), blobs[b]->count());

		len = inBlobs[j]->count();
		Dtype* blob_data = inBlobs[j]->mutable_cpu_data();
		Dtype* minmax = new Dtype[2];//0-min,1-max
		unsigned char* ucharBolbData = new unsigned char[len];
		memset(ucharBolbData,0,len);
		
		//get minmax
		minmax[0] = 1000000.0;
		minmax[1] = -1000000.0;
		for(k = 0; k < len; k++)
		{
			if ( blob_data[k] < minmax[0] )
				minmax[0] = blob_data[k];
			if ( blob_data[k] > minmax[1] )
				minmax[1] = blob_data[k];
		}

		//Normalization
		if ( minmax[0] != minmax[1] )
		{
			tmp = 255.0/(minmax[1] - minmax[0]);
			for(k = 0; k < len; k++)
			{
				ucharBolbData[k] = (unsigned char)((blob_data[k]-minmax[0])*tmp + 0.5) ;
			}
		}

		//write data
        write_real<int>(os, inBlobs[j]->num());
        write_real<int>(os, inBlobs[j]->channels());
        write_real<int>(os, inBlobs[j]->width());
        write_real<int>(os, inBlobs[j]->height());
        write_real<Dtype>(os, minmax, 2);
		write_real<unsigned char>(os, ucharBolbData, len);

		//check
		std::cout<<"len:"<<len<<","<<inBlobs[j]->width()<<"x"<<inBlobs[j]->height()
	      <<"x"<<inBlobs[j]->channels()<<"x"<<inBlobs[j]->num()
	      <<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;

		if (minmax) {delete [] minmax;minmax = 0;}
		if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
	}

}

template <typename Dtype>
 void  Net<Dtype>::blob_uchar2float(std::istream &is, vector<shared_ptr<Blob<Dtype> > > &outBlobs)
{
	//
	int j,k,n,c,w,h;
	size_t len,blob_size;
	Dtype tmp;

	//blob.size
	blob_size = read_real<size_t>(is);
	printf("blob_size:%ld\n",blob_size);
	    
	for(j = 0; j < blob_size; j++){

		//read data
		shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        outBlobs.push_back(blob_pointer);

		n=read_real<int>(is);
        c=read_real<int>(is);
        w=read_real<int>(is);
        h=read_real<int>(is);
        blob_pointer->Reshape(n, c, w, h);
		
		len = n*c*w*h;
		
		
		Dtype* minmax = new Dtype[2];	//0-min,1-max
		unsigned char* ucharBolbData = new unsigned char[len];	//uchar
		Dtype* blob_data = new Dtype[len];	//float
		memset(minmax,0,2);
		memset(ucharBolbData,0,len);
		memset(blob_data,0,len);

		//read data
		read_real<Dtype>(is, minmax, 2);
		read_real<unsigned char>(is, ucharBolbData, len);

		//Normalization
		tmp = (minmax[1] - minmax[0])/255.0;
		for(k = 0; k < len; k++)
		{
			blob_data[k] = (Dtype)( ucharBolbData[k]*tmp + minmax[0] ) ;
		}
		caffe_copy(len, blob_data, blob_pointer->mutable_cpu_data());

		//check
		std::cout<<"len:"<<len<<","<<w<<"x"<<h<<"x"<<c<<"x"<<n<<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;
		
		if (minmax) {delete [] minmax;minmax = 0;}
		if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
		if (blob_data) {delete [] blob_data;blob_data = 0;}
	}

}

template <typename Dtype>
void Net<Dtype>::CopyFrom(const shared_ptr<Net<Dtype> >& other)
{
    std::cout<<"copy net."<<std::endl;
    //string name_;
    name_ = other->name();
    //Phase phase_;
    phase_ = other->phase();
    
    
    //vector<string> layer_names_;
    std::copy(other->layer_names().begin(), other->layer_names().end(), std::back_inserter(layer_names_));
    //map<string, int> layer_names_index_;
    layer_names_index_.insert(other->layer_names_index().begin(), other->layer_names_index().end());
    //vector<string> blob_names_;
    std::copy(other->blob_names().begin(), other->blob_names().end(), std::back_inserter(blob_names_));
    //map<string, int> blob_names_index_;
    blob_names_index_.insert(other->blob_names_index().begin(), other->blob_names_index().end());
    
    //vector<vector<int> > bottom_id_vecs_;
    std::copy(other->bottom_id_vecs().begin(), other->bottom_id_vecs().end(), std::back_inserter(bottom_id_vecs_));
    //vector<vector<int> > top_id_vecs_;
    std::copy(other->top_id_vecs().begin(),other->top_id_vecs().end(), std::back_inserter(top_id_vecs_));
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    //std::copy(other->blobs().begin(), other->blobs().end(), std::back_inserter(blobs_));
    for(int b = 0; b<other->blobs().size(); b++){
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        blobs_.push_back(blob_pointer);
        //blob_pointer->Reshape(other->blobs()[b]->shape());
        blob_pointer->Reshape(other->blobs()[b]->num(), other->blobs()[b]->channels(), other->blobs()[b]->width(), other->blobs()[b]->height());
    }
    
    // -- vector<Blob<Dtype>*> net_input_blobs_;
    //std::copy(other->input_blobs().begin(), other->input_blobs().end(), std::back_inserter(net_input_blobs_));
    net_input_blobs_.push_back(blobs_[0].get());
    
    // -- vector<Blob<Dtype>*> net_output_blobs_;
    //std::copy(other->output_blobs().begin(), other->output_blobs().end(), std::back_inserter(net_output_blobs_));
    net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());
    
    // -- vector<vector<Blob<Dtype>*> > bottom_vecs_;
    //std::copy(other->bottom_vecs().begin(), other->bottom_vecs().end(), std::back_inserter(bottom_vecs_));
    // -- vector<vector<Blob<Dtype>*> > top_vecs_;
    //std::copy(other->top_vecs().begin(), other->top_vecs().end(), std::back_inserter(top_vecs_));
    bottom_vecs_.resize(other->bottom_id_vecs().size());
    top_vecs_.resize(other->top_id_vecs().size());
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    //std::copy(other->layers().begin(), other->layers().end(), std::back_inserter(layers_));
    shared_ptr<Layer<Dtype> > layer;
    for (unsigned int i = 0; i < other->layers_.size(); ++i) {
        //
        layer = other->layers_[i];
        
        //
        for(int b = 0; b<other->bottom_id_vecs()[i].size(); b++)
            bottom_vecs_[i].push_back( blobs_[other->bottom_id_vecs()[i][b]].get() );
        for(int t = 0; t<other->top_id_vecs()[i].size(); t++)
            top_vecs_[i].push_back( blobs_[other->top_id_vecs()[i][t]].get() );
        //
        if( layer->type() == "Split"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "ReLU"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Concat"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Dropout"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Pooling"){            
#ifdef CHECK
			std::cout<<i<<"cp:"<<layer->type()<<":"<<layer->blobs().size()<<std::endl;

			//change
			//PoolingLayer<Dtype>* p = dynamic_cast<PoolingLayer<Dtype>*>(layer.get());

            shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
            sl->CopyFrom(layer->layer_param());
			
			std::cout<<layer->type()<<":SetUp..."<<std::endl;
            sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
			
			std::cout<<layer->type()<<":push_back..."<<std::endl;
            layers_.push_back(sl);
			
			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif
        }
        else if( layer->type() == "Softmax"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Convolution"){
#ifdef CHECK
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;

			//change
            //shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );
            //cl->CopyFrom(layer->layer_param());
            //cl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //cl->CopyBlob(layer);
            
            //layers_.push_back(cl);

			layers_.push_back(layer);
#else 
			layers_.push_back(layer);
#endif
        }
    }
    
    
}


template <typename Dtype>
void Net<Dtype>::CopyFrom_detect(const shared_ptr<Net<Dtype> >& other)
{
    std::cout<<"copy net."<<std::endl;
    //string name_;
    name_ = other->name();
    //Phase phase_;
    phase_ = other->phase();
    
    
    //vector<string> layer_names_;
    std::copy(other->layer_names().begin(), other->layer_names().end(), std::back_inserter(layer_names_));
    //map<string, int> layer_names_index_;
    layer_names_index_.insert(other->layer_names_index().begin(), other->layer_names_index().end());
    //vector<string> blob_names_;
    std::copy(other->blob_names().begin(), other->blob_names().end(), std::back_inserter(blob_names_));
    //map<string, int> blob_names_index_;
    blob_names_index_.insert(other->blob_names_index().begin(), other->blob_names_index().end());
    
    //vector<vector<int> > bottom_id_vecs_;
    std::copy(other->bottom_id_vecs().begin(), other->bottom_id_vecs().end(), std::back_inserter(bottom_id_vecs_));
    //vector<vector<int> > top_id_vecs_;
    std::copy(other->top_id_vecs().begin(),other->top_id_vecs().end(), std::back_inserter(top_id_vecs_));
    
    // -- vector<shared_ptr<Blob<Dtype> > > blobs_;
    //std::copy(other->blobs().begin(), other->blobs().end(), std::back_inserter(blobs_));
    for(int b = 0; b<other->blobs().size(); b++){
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        blobs_.push_back(blob_pointer);
        //blob_pointer->Reshape(other->blobs()[b]->shape());
        blob_pointer->Reshape(other->blobs()[b]->num(), other->blobs()[b]->channels(), other->blobs()[b]->width(), other->blobs()[b]->height());
    }
    
    // -- vector<Blob<Dtype>*> net_input_blobs_;
    //std::copy(other->input_blobs().begin(), other->input_blobs().end(), std::back_inserter(net_input_blobs_));
    net_input_blobs_.push_back(blobs_[0].get());	//input: "data"
	net_input_blobs_.push_back(blobs_[1].get());	//input: "im_info"
    
    // -- vector<Blob<Dtype>*> net_output_blobs_;
    //std::copy(other->output_blobs().begin(), other->output_blobs().end(), std::back_inserter(net_output_blobs_));
	net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());		//top: "cls_prob",Softmax
	net_output_blobs_.push_back(blobs_[blobs_.size()-2].get());		//top: "bbox_pred",
	net_output_blobs_.push_back(blobs_[blobs_.size()-12].get());	//top: "rois",ProposalLayer;"Split Layer"
    
    // -- vector<vector<Blob<Dtype>*> > bottom_vecs_;
    //std::copy(other->bottom_vecs().begin(), other->bottom_vecs().end(), std::back_inserter(bottom_vecs_));
    // -- vector<vector<Blob<Dtype>*> > top_vecs_;
    //std::copy(other->top_vecs().begin(), other->top_vecs().end(), std::back_inserter(top_vecs_));
    bottom_vecs_.resize(other->bottom_id_vecs().size());
    top_vecs_.resize(other->top_id_vecs().size());
    
    // -- vector<shared_ptr<Layer<Dtype> > > layers_;
    //std::copy(other->layers().begin(), other->layers().end(), std::back_inserter(layers_));
    shared_ptr<Layer<Dtype> > layer;
    for (unsigned int i = 0; i < other->layers_.size(); ++i) {
        //
        layer = other->layers_[i];
        
        //
        for(int b = 0; b<other->bottom_id_vecs()[i].size(); b++)
            bottom_vecs_[i].push_back( blobs_[other->bottom_id_vecs()[i][b]].get() );
        for(int t = 0; t<other->top_id_vecs()[i].size(); t++)
            top_vecs_[i].push_back( blobs_[other->top_id_vecs()[i][t]].get() );
        //
        if( layer->type() == "Split"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
            //shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "ReLU"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
            //shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Concat"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
            //shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Dropout"){
            //std::cout<<"cp:"<<layer->type()<<std::endl;
            //shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Pooling"){    
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
#ifdef CHECK
            shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
            sl->CopyFrom(layer->layer_param());
			
			std::cout<<layer->type()<<":SetUp..."<<std::endl;
            sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
			
			std::cout<<layer->type()<<":push_back..."<<std::endl;
            layers_.push_back(sl);
			
			std::cout<<layer->type()<<":end!!"<<std::endl;

#else 
			layers_.push_back(layer);
#endif
        }
        else if( layer->type() == "Softmax"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
            //shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
            //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
            //layers_.push_back(sl);

			layers_.push_back(layer);
        }
        else if( layer->type() == "Convolution"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
#ifdef CHECK
			//change
            shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
            cl->CopyFrom(layer->layer_param());

			std::cout<<layer->type()<<":SetUp..."<<std::endl;
            cl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);

			std::cout<<layer->type()<<":CopyBlob..."<<std::endl;
            cl->CopyBlob(layer);

			std::cout<<layer->type()<<":push_back..."<<std::endl;
            layers_.push_back(cl);

			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif
        }
		else if( layer->type() == "Reshape"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
			
#ifdef CHECK
            shared_ptr<ReshapeLayer<Dtype> > sl(new ReshapeLayer<Dtype>( ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
            sl->CopyFrom(layer->layer_param());
			
			std::cout<<layer->type()<<":SetUp..."<<std::endl;
            sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
			
			std::cout<<layer->type()<<":push_back..."<<std::endl;
            layers_.push_back(sl);
			
			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif
        }
        else if( layer->type() == "Deconvolution"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
#ifdef CHECK
			//change
			shared_ptr<DeconvolutionLayer<Dtype> > cl(new DeconvolutionLayer<Dtype>(  ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
			cl->CopyFrom(layer->layer_param());

			std::cout<<layer->type()<<":SetUp..."<<std::endl;
			cl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);

			std::cout<<layer->type()<<":CopyBlob..."<<std::endl;
			cl->CopyBlob(layer);

			std::cout<<layer->type()<<":push_back..."<<std::endl;
			layers_.push_back(cl);

			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif
        }
		else if( layer->type() == "ProposalLayer"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
						
#ifdef CHECK
			shared_ptr<ProposalLayer<Dtype> > sl(new ProposalLayer<Dtype>( ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
			sl->CopyFrom(layer->layer_param());
			
			std::cout<<layer->type()<<":SetUp..."<<std::endl;
			sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
			
			std::cout<<layer->type()<<":push_back..."<<std::endl;
			layers_.push_back(sl);
						
			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif

        }
		else if( layer->type() == "ROIPooling"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
			
#ifdef CHECK
			shared_ptr<ROIPoolingLayer<Dtype> > sl(new ROIPoolingLayer<Dtype>( ) );

			std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
			sl->CopyFrom(layer->layer_param());
			
			std::cout<<layer->type()<<":SetUp..."<<std::endl;
			sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
			
			std::cout<<layer->type()<<":push_back..."<<std::endl;
			layers_.push_back(sl);
			
			std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
			layers_.push_back(layer);
#endif

        }
		else if( layer->type() == "InnerProduct"){
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
#ifdef CHECK
		//change
		shared_ptr<InnerProductLayer<Dtype> > cl(new InnerProductLayer<Dtype>(  ) );

		std::cout<<layer->type()<<":CopyFrom..."<<std::endl;
		cl->CopyFrom(layer->layer_param());

		std::cout<<layer->type()<<":SetUp..."<<std::endl;
		cl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);

		std::cout<<layer->type()<<":CopyBlob..."<<std::endl;
		cl->CopyBlob(layer);

		std::cout<<layer->type()<<":push_back..."<<std::endl;
		layers_.push_back(cl);

		std::cout<<layer->type()<<":end!!"<<std::endl;
#else 
		layers_.push_back(layer);
#endif

        }
		else
		{
			std::cout<<i<<" cp:"<<layer->type()<<std::endl;
			layers_.push_back(layer);
		}
    }
    
    
}
    
INSTANTIATE_CLASS(Net);

}  // namespace caffe
