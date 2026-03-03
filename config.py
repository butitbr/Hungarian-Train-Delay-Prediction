gtfs_download_link = 'https://{username}:{password}@www.mavcsoport.hu/gtfs/gtfsMavMenetrend.zip'
data_root = "data/"
gtfs_folder = data_root + "gtfsMavMenetrend/"
gtfs_stops_file = "stops.txt"
gtfs_dowload_location = data_root + "gtfs/"
passanger_info_folder = data_root + '2_147_utastajekoztatas_'
years_to_process = [2022, 2023]
event_file_name = "vonat_esemenyek_"
delay_causes_dir = data_root + "kesesi_ok/"
cache_location=data_root+'cache/'
generated_files_path=data_root + 'generated_files/'
weather_folder = generated_files_path + "odp/"
model_location="model/"

# OLD API (deprecated): "https://vonatinfo.mav-start.hu/map.aspx/getData"
# NEW GraphQL API:
mav_api_url = "https://mavplusz.hu/otp2-backend/otp/routers/default/index/graphql"

# Legacy hardcoded data (kept for backwards compatibility)
# ezeken az állomásokon nézzük meg, milyen vonatok közlekednek
# https://hu.wikipedia.org/wiki/Budapest%E2%80%93Esztergom-vas%C3%BAtvonal
# https://hu.wikipedia.org/wiki/Kiskunf%C3%A9legyh%C3%A1za%E2%80%93Orosh%C3%A1za-vas%C3%BAtvonal

main_stations = {2: ["Budapest-Nyugati",
                     "Újpest",
                     "Pilisvörösvár",
                     "Piliscsaba",
                     "Solymár",
                     "Üröm",
                     "Dorog",
                     "Esztergom"],
                 147: ["Kiskunfélegyháza", "Gátér", "Csongrád", "Szentes", "Fábiánsebestyén", "Gádoros",
                       "Orosháza"]
                 }

collected_trains = {
    2: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,
        2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045,
        2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2060, 2061, 2062, 2063, 2064,
        2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2080, 2081, 2082, 2083,
        2084, 2085, 2086, 2087, 2088, 2090, 2103, 2117, 2127, 2171, 2178, 2181, 2188, 2193, 2203, 2204, 2282, 2294,
        3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3226, 3227, 3228,
        3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246,
        3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264,
        3265, 3266, 3267, 3268, 3269, 3270, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3280, 3281, 3282, 3283, 3284,
        3285, 3286, 3287],
    147: [17243, 32610, 32612, 32613, 32614, 32615, 32616, 32617, 32619, 32626, 37410, 37412, 37414, 37415, 37416,
          37417, 37418, 37419, 37421, 37428, 37431, 37435, 37436, 37438, 37439, 37452, 37454, 37455, 37456, 37457,
          37461, 37463, 37465, 37466, 37467]
}
# ?
not_existing = [2069, 2070, 2080, 2090, 2103, 2117, 2127, 2193, 2203, 2204, 2282, 2294, 3250, 3260, 3270, 3280, 3281,
                32610, 32613, 37410, 37416, 37419, 37435, 37438, 37439, 37456, 37457, 37461, 37465]
