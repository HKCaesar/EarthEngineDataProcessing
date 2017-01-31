import ogr, gdal
import numpy as np


def read_land_use(da_shapefile="SDM324649_full/ll_gda94/sde_shape/whole/VIC/CATCHMENTS/layer/landuse_2014.shp",
                  resolution=(1022, 973),
                  area_filter="POLYGON ((143.32317350376297 -37.496296386368165, 143.32180000642074 -37.70330610816869,\
                        143.59543478539388 -37.70317028876007, 143.5968081449812 -37.49575155973978,\
                        143.32317350376297 -37.496296386368165))",
                  buffer=0):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(da_shapefile, 0)
    if data_source is None:
        print('Could not open %s' % da_shapefile)
        return [], {}
    else:
        print('Opened %s' % da_shapefile)
        layer = data_source.GetLayer()
        feature_count = layer.GetFeatureCount()
        print("Number of features: %d" % feature_count)

    filter_poly = ogr.CreateGeometryFromWkt(area_filter)
    layer.SetSpatialFilter(filter_poly)

    selected_feature_count = layer.GetFeatureCount()
    print("Number of features in selected area: %d" % selected_feature_count)

    unique_classes = np.unique([feature.items()['LC_DESC_14'] for feature in layer])
    unique_classes_dict = {i + 1: unique_classes[i] for i in range(len(unique_classes))}
    unique_classes_dict[0] = 'No data'

    no_data_value = 0
    layer.SetAttributeFilter(None)
    x_min, x_max, y_min, y_max = filter_poly.GetEnvelope()
    # x_res = 1022
    # y_res = 973
    x_res, y_res = resolution
    pixel_size = (x_max - x_min) / x_res

    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    # target_ds.SetGeoTransform((x_max, pixel_size, 0, y_min, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    empty_array = np.ones((y_res, x_res)) * no_data_value
    band.WriteArray(empty_array)

    for i, cls in unique_classes_dict.items():
        if i != 0:
            layer.SetAttributeFilter("LC_DESC_14 = '%s'" % cls)
            gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[i])

    layer.SetAttributeFilter(None)
    array = band.ReadAsArray()

    return array, unique_classes_dict
