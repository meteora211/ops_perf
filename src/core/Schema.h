#pragma once
#include <string>

// TODO: use function schema instead of string
class Schema {
public:
  Schema() = default;
  ~Schema() = default;

  Schema(std::string name) : name_(name){}
  // Schema(Schema &&schema)
  //     : name_(std::move(schema.name_)),
  //       signature_(std::move(schema.signature_)) {}

  std::string name() const { return name_; }
  std::string signature() const { return signature_; }

private:
  std::string name_;
  std::string signature_;
};

inline bool operator==(const Schema& lhs, const Schema& rhs) {
  return lhs.name() == rhs.name() && lhs.signature() == rhs.signature();
}

inline bool operator!=(const Schema& lhs, const Schema& rhs) {
  return !operator==(lhs, rhs);
}

namespace std {
  template <>
  struct hash<Schema> {
    size_t operator()(const Schema& s) const {
      return std::hash<std::string>()(s.name()) ^ (~ std::hash<std::string>()(s.signature()));
    }
  };
}
