#ifndef AM_QADF_NATIVE_QUERY_CT_IMAGE_READER_HPP
#define AM_QADF_NATIVE_QUERY_CT_IMAGE_READER_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <string>
#include <memory>

#ifdef ITK_AVAILABLE
// Forward declaration for ITK types (avoid including ITK headers in header)
namespace itk {
    template<typename TPixel, unsigned int VImageDimension>
    class Image;
}
#endif

namespace am_qadf_native {
namespace query {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// CT image reader using ITK
class CTImageReader {
public:
    // Read DICOM series from directory
    FloatGridPtr readDICOMSeries(const std::string& directory);
    
    // Read NIfTI file
    FloatGridPtr readNIfTI(const std::string& filename);
    
    // Read TIFF stack
    FloatGridPtr readTIFFStack(const std::string& directory);
    
    // Read raw binary file
    FloatGridPtr readRawBinary(
        const std::string& filename,
        int width, int height, int depth,
        float voxel_size = 1.0f
    );
    
private:
    // Convert ITK image to OpenVDB grid (legacy void* interface)
    FloatGridPtr convertITKToOpenVDB(void* itk_image);
    
#ifdef ITK_AVAILABLE
    // Specialized helper for Image<float, 3>
    // Note: Implementation uses full ITK types in .cpp file
    FloatGridPtr convertITKImageToOpenVDB(void* itkImage);
#endif
};

} // namespace query
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_QUERY_CT_IMAGE_READER_HPP
