// Copyright 2015, Additive Regularization of Topic Models.

#include "artm/core/processor_input.h"

namespace artm {
namespace core {

CacheManager* ProcessorInput::cache_manager(CacheManager_Type type) const {
  for (auto iter = cache_manager_.begin(); iter != cache_manager_.end(); ++iter)
    if (iter->first == type)
      return iter->second;
  return nullptr;
}

void ProcessorInput::set_cache_manager(CacheManager_Type type, CacheManager* cache_manager) {
  for (auto iter = cache_manager_.begin(); iter != cache_manager_.end(); ++iter) {
    if (iter->first == type) {
      iter->second = cache_manager;
      return;
    }
  }

  cache_manager_.push_back(std::make_pair(type, cache_manager));
}

bool ProcessorInput::has_cache_manager(CacheManager_Type type) const {
  return cache_manager(type) != nullptr;
}

}  // namespace core
}  // namespace artm
