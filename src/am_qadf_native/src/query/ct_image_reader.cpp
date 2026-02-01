#include "am_qadf_native/query/ct_image_reader.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <stdexcept>

#ifdef ITK_AVAILABLE
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageIOBase.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#endif

namespace am_qadf_native {
namespace query {

#ifdef ITK_AVAILABLE
// Forward declaration
FloatGridPtr convertITKImageToOpenVDBImpl(itk::Image<float, 3>::Pointer itkImage);
#endif

FloatGridPtr CTImageReader::readDICOMSeries(const std::string& directory) {
    openvdb::initialize();
    
#ifdef ITK_AVAILABLE
    try {
        // Define ITK image type (3D float image)
        using ImageType = itk::Image<float, 3>;
        using ReaderType = itk::ImageSeriesReader<ImageType>;
        using ImageIOType = itk::GDCMImageIO;
        using NamesGeneratorType = itk::GDCMSeriesFileNames;
        
        // Create DICOM series file names generator
        NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
        nameGenerator->SetInputDirectory(directory);
        
        // Get series UIDs (unique identifiers for DICOM series)
        using SeriesIdContainer = std::vector<std::string>;
        const SeriesIdContainer& seriesUIDs = nameGenerator->GetSeriesUIDs();
        
        if (seriesUIDs.empty()) {
            throw std::runtime_error("No DICOM series found in directory: " + directory);
        }
        
        // Use the first series (can be extended to select specific series)
        std::string seriesIdentifier = seriesUIDs.begin()->c_str();
        
        // Get file names for this series
        using FileNamesContainer = std::vector<std::string>;
        FileNamesContainer fileNames = nameGenerator->GetFileNames(seriesIdentifier);
        
        if (fileNames.empty()) {
            throw std::runtime_error("No DICOM files found for series: " + seriesIdentifier);
        }
        
        // Create reader
        ReaderType::Pointer reader = ReaderType::New();
        ImageIOType::Pointer dicomIO = ImageIOType::New();
        reader->SetImageIO(dicomIO);
        reader->SetFileNames(fileNames);
        
        // Read the image
        reader->Update();
        ImageType::Pointer itkImage = reader->GetOutput();
        
        // Convert ITK image to OpenVDB grid
        return convertITKImageToOpenVDBImpl(itkImage);
        
    } catch (const std::exception& e) {
        // Return empty grid on error
        return FloatGrid::create();
    }
#else
    // ITK not available - return empty grid
    return FloatGrid::create();
#endif
}

FloatGridPtr CTImageReader::readNIfTI(const std::string& filename) {
    openvdb::initialize();
    
#ifdef ITK_AVAILABLE
    try {
        // Define ITK image type (3D float image)
        using ImageType = itk::Image<float, 3>;
        using ReaderType = itk::ImageFileReader<ImageType>;
        
        // Create reader
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName(filename);
        
        // Read the image
        reader->Update();
        ImageType::Pointer itkImage = reader->GetOutput();
        
        // Convert ITK image to OpenVDB grid
        return convertITKImageToOpenVDBImpl(itkImage);
        
    } catch (const std::exception& e) {
        // Return empty grid on error
        return FloatGrid::create();
    }
#else
    // ITK not available - return empty grid
    return FloatGrid::create();
#endif
}

FloatGridPtr CTImageReader::readTIFFStack(const std::string& directory) {
    // TODO: Implement TIFF stack reading
    // Similar to DICOM but using TIFF file reader
    openvdb::initialize();
    return FloatGrid::create();
}

FloatGridPtr CTImageReader::readRawBinary(
    const std::string& filename,
    int width, int height, int depth,
    float voxel_size
) {
    openvdb::initialize();
    
    try {
        // Open file
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // Create OpenVDB grid
        auto grid = FloatGrid::create();
        grid->setTransform(
            openvdb::math::Transform::createLinearTransform(voxel_size)
        );
        
        // Read binary data and populate grid
        auto& tree = grid->tree();
        const size_t total_voxels = static_cast<size_t>(width) * height * depth;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float value = 0.0f;
                    file.read(reinterpret_cast<char*>(&value), sizeof(float));
                    
                    if (file.eof()) {
                        break;
                    }
                    
                    // Only store non-zero values (sparse storage)
                    if (value != 0.0f) {
                        openvdb::Coord coord(x, y, z);
                        tree.setValue(coord, value);
                    }
                }
            }
        }
        
        file.close();
        return grid;
        
    } catch (const std::exception& e) {
        // Return empty grid on error
        auto grid = FloatGrid::create();
        grid->setTransform(
            openvdb::math::Transform::createLinearTransform(voxel_size)
        );
        return grid;
    }
}

#ifdef ITK_AVAILABLE
// Helper function to convert ITK image to OpenVDB grid
// Specialized for Image<float, 3>
FloatGridPtr convertITKImageToOpenVDBImpl(itk::Image<float, 3>::Pointer itkImage) {
    using ImageType = itk::Image<float, 3>;
    
    // Get image properties
    typename ImageType::RegionType region = itkImage->GetLargestPossibleRegion();
    typename ImageType::SpacingType spacing = itkImage->GetSpacing();
    
    // Create OpenVDB grid
    auto grid = FloatGrid::create();
    
    // Set transform (voxel size from ITK spacing)
    // Use average spacing (assuming uniform spacing)
    float voxel_size = static_cast<float>((spacing[0] + spacing[1] + spacing[2]) / 3.0);
    if (voxel_size <= 0.0f) {
        voxel_size = 1.0f;  // Default
    }
    
    grid->setTransform(
        openvdb::math::Transform::createLinearTransform(voxel_size)
    );
    
    // Copy data from ITK image to OpenVDB grid
    auto& tree = grid->tree();
    
    // Use iterator to traverse ITK image
    itk::ImageRegionConstIterator<ImageType> it(itkImage, region);
    it.GoToBegin();
    
    while (!it.IsAtEnd()) {
        float value = it.Get();
        
        // Get index in ITK image
        typename ImageType::IndexType index = it.GetIndex();
        
        // Only store non-zero values (sparse storage)
        if (value != 0.0f) {
            openvdb::Coord coord(
                static_cast<int>(index[0]),
                static_cast<int>(index[1]),
                static_cast<int>(index[2])
            );
            tree.setValue(coord, value);
        }
        
        ++it;
    }
    
    return grid;
}
#endif

FloatGridPtr CTImageReader::convertITKToOpenVDB(void* itk_image_ptr) {
    openvdb::initialize();
    
#ifdef ITK_AVAILABLE
    if (!itk_image_ptr) {
        return FloatGrid::create();
    }
    
    // Cast void pointer to ITK image pointer
    // Assuming Image<float, 3> type
    using ImageType = itk::Image<float, 3>;
    ImageType::Pointer itkImage = static_cast<ImageType*>(itk_image_ptr);
    
    return convertITKImageToOpenVDBImpl(itkImage);
#else
    return FloatGrid::create();
#endif
}

#ifdef ITK_AVAILABLE
// Specialized helper for Image<float, 3>
FloatGridPtr CTImageReader::convertITKImageToOpenVDB(void* itkImagePtr) {
    if (!itkImagePtr) {
        return FloatGrid::create();
    }
    
    // Cast to ITK image pointer
    using ImageType = itk::Image<float, 3>;
    ImageType::Pointer itkImage = static_cast<ImageType*>(itkImagePtr);
    
    return convertITKImageToOpenVDBImpl(itkImage);
}
#endif

} // namespace query
} // namespace am_qadf_native
