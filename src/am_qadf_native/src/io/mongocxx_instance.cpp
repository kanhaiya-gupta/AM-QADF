#include "am_qadf_native/io/mongocxx_instance.hpp"

namespace am_qadf_native {
namespace io {

mongocxx::instance& get_mongocxx_instance() {
    static mongocxx::instance instance{};
    return instance;
}

}  // namespace io
}  // namespace am_qadf_native
