REGISTER_OP(add, Tensor(const Tensor&, const Tensor&), CPU, core::add_tensor_cpu)
REGISTER_OP(ones, Tensor(std::vector<int64_t>, ScalarType), CPU, core::ones_cpu)
REGISTER_OP(empty, Tensor(const std::vector<int64_t>&, ScalarType), CPU, core::empty_cpu)
