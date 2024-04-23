#include <TensorImpl.h>
#include <Storage.h>

namespace core {

TensorImpl::TensorImpl(Storage &&storage, ScalarType dtype)
    : data_type_(dtype), numel_(1), storage_(std::move(storage)) {}

int64_t TensorImpl::numel() const { return numel_; }

} // namespace core
