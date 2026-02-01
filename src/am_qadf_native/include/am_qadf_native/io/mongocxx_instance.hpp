#ifndef AM_QADF_NATIVE_IO_MONGOCXX_INSTANCE_HPP
#define AM_QADF_NATIVE_IO_MONGOCXX_INSTANCE_HPP

#include <mongocxx/instance.hpp>

namespace am_qadf_native {
namespace io {

/** Single process-wide mongocxx::instance. Call before using any mongocxx API. */
mongocxx::instance& get_mongocxx_instance();

}  // namespace io
}  // namespace am_qadf_native

#endif  // AM_QADF_NATIVE_IO_MONGOCXX_INSTANCE_HPP
