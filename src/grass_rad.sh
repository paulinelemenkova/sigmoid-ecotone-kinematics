#!/usr/bin/env bash
set -euo pipefail
DEM=$1
OUT=$2
DAYS=${DAYS:-"$(seq -s " " 152 5 272)"}
mkdir -p "$OUT"
r.in.gdal -o input="$DEM" output=dem --quiet
g.region raster=dem
r.null map=dem setnull=-9999 --quiet
g.region raster=dem res=90 -a
r.resamp.stats input=dem output=dem90 method=average --quiet
r.slope.aspect elevation=dem90 slope=slope90 aspect=aspect90 --quiet
maps=""
for d in $DAYS; do
    r.sun elevation=dem90 slope=slope90 aspect=aspect90 day="$d" linke_value=3.0 albedo_value=0.2 \
        glob_rad="glob_$d" nprocs=2 --quiet
    maps="$maps,glob_$d"
done
r.series input="${maps#,}" output=rad_jjas method=average --quiet
r.mapcalc "rad_i = round(rad_jjas)" --quiet
r.mapcalc "dem90_i = round(dem90)" --quiet
r.out.gdal -c -f input=rad_i output="$OUT/rad_jjas_90m.tif" type=UInt16 createopt="COMPRESS=DEFLATE,PREDICTOR=2" --quiet
r.out.gdal -c -f input=dem90_i output="$OUT/dem_90m.tif" type=Int16 createopt="COMPRESS=DEFLATE,PREDICTOR=2" --quiet
r.univar -g map=rad_jjas | grep -E '^(min|max|mean)='
