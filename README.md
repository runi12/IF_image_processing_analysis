# IF_image_processing_analysis
These are scripts made to analyse fluorescence assays. Values for threhsolding or filtering were determined empyrically, so recalibration may be required if you intend to use these scripts in your own work. Results of these analyses can be found in the paper "Oncolytic Effect of Zika Virus in Glioblastoma is potentiated by low MGMT expression and mirrors Temozolomide response".

# yH2AX
The focus of this analysis was to determine DNA damage after treatment with ZIKA virus and/or an MGMT inhibtor. DNA lesions were marked with anti-yH2AX antibodies. The staining pattern is typically dots (red channel) spread across the nucleus (blue channel), however in dead cells, the dots can also be found in the cytoplasm.

# Live and Dead (2D and 3D)
The focus of this analysis was to observe cytotoxicity in Wide-Field microscopy images. The Live and Dead kit can stain the cytoplasm of dead cells with PI (typically in red channel) and live cells with Calcein (typically in green). Processing and measurements were made with classic vision, and contour detection.
