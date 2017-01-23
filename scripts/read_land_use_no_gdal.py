import numpy as np
import fiona
from shapely.geometry import Polygon
from shapely.wkt import dumps, loads
from shapely.geometry import asShape, mapping
from rasterio import features, transform


def read_land_use(da_shapefile="SDM324649_full/ll_gda94/sde_shape/whole/VIC/CATCHMENTS/layer/landuse_2014.shp",
                  resolution=(1022, 973),
                  area_filter="POLYGON ((143.32317350376297 -37.496296386368165, 143.32180000642074 -37.70330610816869,\
                        143.59543478539388 -37.70317028876007, 143.5968081449812 -37.49575155973978,\
                        143.32317350376297 -37.496296386368165))",
                  buffer=0):
    file = fiona.open(da_shapefile)
    filter_poly = loads(area_filter)
    filtered = list(file.values(bbox=filter_poly.bounds))

    for feature in filtered:
        geom = asShape(feature['geometry'])
        buffered = geom.buffer(buffer)
        #     print(buffered)
        if buffered.is_empty:
            filtered.remove(feature)
        else:
            feature['geometry'] = mapping(buffered)

    unique_classes = np.unique([feature['properties']['LC_DESC_14'] for feature in filtered])
    unique_classes_dict = {i + 1: unique_classes[i] for i in range(len(unique_classes))}
    unique_classes_dict[0] = 'No data'
    unique_classes_inv = {v: k for k, v in unique_classes_dict.items()}
    shapes = ((feature['geometry'], unique_classes_inv[feature['properties']['LC_DESC_14']]) for feature in filtered)
    x_min, y_min, x_max, y_max = filter_poly.bounds
    x_res, y_res = resolution
    # pixel_size = (x_max - x_min) / x_res
    image = features.rasterize(
        ((g, v) for g, v in shapes),
        out_shape=(y_res, x_res),
        transform=transform.from_bounds(x_min, y_min, x_max, y_max, x_res, y_res))
        #transform=[x_min, pixel_size, 0, y_max, 0, -pixel_size])
    return image, unique_classes_dict
