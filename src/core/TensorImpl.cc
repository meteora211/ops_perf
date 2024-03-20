#include <TensorImpl.h>
#include <Storage.h>

namespace core {

TensorImpl::TensorImpl(Storage &&storage)
    : numel_(0), storage_(std::move(storage)) {}

int64_t TensorImpl::numel() const { return numel_; }

} // namespace core
