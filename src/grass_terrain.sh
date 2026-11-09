#!/usr/bin/env bash
set -euo pipefail
DEM=$1
OUT=$2
DAYS=${DAYS:-"$(seq -s " " 152 5 272)"}
mkdir -p "$OUT"
t0=$(date +%s)
el() { echo "  [$(($(date +%s) - t0)) s] $*"; }

r.in.gdal -o input="$DEM" output=dem --quiet
g.region raster=dem
r.null map=dem setnull=-9999 --quiet
el "imported: $(g.region -g | grep -E '^(rows|cols)=' | tr '\n' ' ')"

r.slope.aspect elevation=dem slope=slope aspect=aspect_ccw --quiet
r.slope.aspect -n elevation=dem aspect=aspect_cw --quiet
r.mapcalc "northness = if(aspect_cw < 0, 0, cos(aspect_cw))" --quiet
el "slope/aspect done"

r.geomorphon elevation=dem forms=geomorph search=25 skip=3 flat=2 --quiet
el "geomorphon done"

g.region raster=dem res=90 -a
r.resamp.stats input=dem output=dem90 method=average --quiet
r.slope.aspect elevation=dem90 slope=slope90 aspect=aspect90 --quiet
maps=""
for d in $DAYS; do
    r.sun elevation=dem90 slope=slope90 aspect=aspect90 day="$d" linke_value=3.0 albedo_value=0.2 \
        glob_rad="glob_$d" nprocs=2 --quiet
    maps="$maps,glob_$d"
    el "r.sun day $d"
done
r.series input="${maps#,}" output=rad_jjas method=average --quiet
el "rad_jjas done"

g.region raster=dem
r.mapcalc "slope_deg = round(slope)" --quiet
r.mapcalc "aspect_deg = if(isnull(aspect_cw) || aspect_cw < 0, 0, round(aspect_cw))" --quiet
r.out.gdal -c input=geomorph output="$OUT/geomorph_30m.tif" type=Byte createopt="COMPRESS=DEFLATE" nodata=0 --quiet
r.out.gdal -c -f input=slope_deg output="$OUT/slope_30m.tif" type=Byte createopt="COMPRESS=DEFLATE" --quiet
r.out.gdal -c -f input=aspect_deg output="$OUT/aspect_30m.tif" type=UInt16 createopt="COMPRESS=DEFLATE" --quiet
g.region raster=dem90
r.mapcalc "rad_i = round(rad_jjas)" --quiet
r.mapcalc "dem90_i = round(dem90)" --quiet
r.out.gdal -c -f input=rad_i output="$OUT/rad_jjas_90m.tif" type=UInt16 createopt="COMPRESS=DEFLATE,PREDICTOR=2" --quiet
r.out.gdal -c -f input=dem90_i output="$OUT/dem_90m.tif" type=Int16 createopt="COMPRESS=DEFLATE,PREDICTOR=2" --quiet
el "exported"
r.univar -g map=rad_jjas | grep -E '^(min|max|mean|stddev)='
