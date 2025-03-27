import shutil

import dotenv

from config import data_root,weather_folder,gtfs_folder,gtfs_stops_file,passanger_info_folder,years_to_process,mav_api_url
import pandas as pd
import requests
import json
from io import StringIO
import re
import datetime
import numpy as np
import os

dotenv.load_dotenv()
owm_key=os.environ['openweathermap_api_key']


def reload_dotenv():
    env_vars = dotenv.dotenv_values(".env")
    for key, value in env_vars.items():
        os.environ[key] = value

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory '{folder_name}' created successfully!")
    else:
        print(f"Directory '{folder_name}' already exists.")


def download_gtfs_zip():
    dotenv.load_dotenv()
    today_str = datetime.datetime.today().strftime('%Y-%m-%d')
    from config import gtfs_download_link, gtfs_dowload_location
    create_folder_if_not_exists(gtfs_dowload_location)
    username = os.getenv('gtfs_user')
    password = os.getenv('gtfs_pw')
    with open(gtfs_dowload_location + today_str + '_gtfs.zip', 'wb') as file:
        response = requests.get(gtfs_download_link.format(username=username, password=password), stream=True)
        print('Response Code:', response.status_code)
        file.write(response.content)
    latest_folder = gtfs_dowload_location + "latest/"
    create_folder_if_not_exists(latest_folder)
    shutil.copyfile(gtfs_dowload_location + today_str + '_gtfs.zip', latest_folder+"gtfs.zip")

'''data_root="data/"
weather_folder=data_root+"odp/"
gtfs_folder=data_root+"gtfsMavMenetrend/"
gtfs_stops_file="stops.txt"
'''
def get_location_data():
    gtfs_stops=pd.read_csv(gtfs_folder+gtfs_stops_file)
    locs=[]
    passanger_info_folders=[passanger_info_folder+str(y)+"/" for y in years_to_process]
    for l in passanger_info_folders:
        passanger_info_locations_file=l+"t_szolg_helyek.txt"
        places_=pd.read_csv(passanger_info_locations_file,sep=',',encoding='iso-8859-2')
        locs.append(places_)
    places=pd.concat(locs,axis=0)
    for l in locs:
        del l
    #TODO check multiple appearence
    places=places.groupby('TELJES_NEV').agg("last")
    places_with_gtfs=places.merge(gtfs_stops,how='left', left_on='POLGARI_NEV', right_on='stop_name')
    return places_with_gtfs


# Egy állomás napi menetrendjének lehívása
def pull_station_data_from_API(station_name):
    station_query = {
        "a": "STATION", "jo": {
            "a": f"{station_name}"
        }
    }
    response = requests.post(mav_api_url, json=station_query)
    return response


def extract_plus_info(df):
    info = []
    columns_to_drop = []
    for c in df.columns:
        if 'Unnamed' in c[0]:
            info.append(c[1])
            full_nan = df[c].isnull().all()
            if full_nan:
                columns_to_drop.append(c)
            print('Full nan col?', c, full_nan)

    return df.drop(columns=columns_to_drop), ' - '.join(list(set(info)))

def pull_train_data_from_API(train_no):
    train_query={
      "a": "TRAIN",
      "jo": {
        "vsz": str(train_no),
        "zoom": True
      }
    }
    response = requests.post(mav_api_url, json=train_query)
    return response

def process_train(erk,ind):
    new_cols_trains = ['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']
    # ez nem jó, a tegnap indultak közlekedési napja tegnap!!
    # d=datetime.date.today()
    erk_teny,erk_terv,ind_teny,ind_terv=process_plan_fact_time_cols(erk, ind)
    return pd.Series(dict(zip(new_cols_trains,[erk_terv,erk_teny,ind_terv,ind_teny])))
def process_train_desc_t(train_desc):
    train_desc.split()
    # Regular expression to find text within parentheses
    pattern = r'\((.*?)\)'
    # Find all matches
    matches = re.findall(pattern, train_desc)
    train_desc_2 = matches[0]
    train_desc_2_parts=train_desc_2.split(", ")
    date=pd.to_datetime(train_desc_2_parts[1]).date()
    start_station=train_desc_2_parts[0].split(" - ")[0]
    end_station=train_desc_2_parts[0].split(" - ")[1]
    train_desc_1=train_desc.replace("("+train_desc_2+")", "")
    train_desc_1_parts=train_desc_1.split()
    train_no=train_desc_1_parts[0]
    train_name=train_desc_1_parts[-1]
    train_type=' '.join(train_desc_1_parts[1:-1])
    if train_type.isupper():
        t=train_type
        train_type=train_name
        train_name=t
    return date,train_no,train_name,start_station,end_station,train_type

def get_train_data(train_no):
    print('-> Pulling ', train_no)
    # ez kell hogy lássok mi a tény és mi az előrejelzés, kellhet
    #now = datetime.now().time()
    train_no_l = train_no
    if not str(train_no).startswith('55'):
        train_no_l = int('55' + str(train_no))

    train_resp = pull_train_data_from_API(train_no_l)
    train_dict = json.loads(train_resp.text)
    try:
        train_df = pd.read_html(StringIO(train_dict['d']['result']['html']))[0]
    except:
        print('----------------------------- Failed', train_no)
        # failed[train_no] = train_dict
        return None
    train_df, info = extract_plus_info(train_df)
    #print(info)

    #print(train_df.columns)
    train_desc = train_df.columns[0][0]
    date, train_no, train_name, start_station, end_station, train_type = process_train_desc_t(train_desc)
    new_col_dict = {c: c[-1] for c in train_df.columns}
    #print(new_col_dict)
    train_df.columns = new_col_dict.values()
    train_data = pd.concat(
        [train_df, train_df.apply(lambda x: process_train(x["Érk."], x["Ind."]), axis=1)],
        axis=1)
    #print(train_data.columns)
    #print(train_data.index)

    train_data[['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']] = train_data[['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']].map(
        lambda x: pd.to_datetime(str(date)+' '+x, format='%Y-%m-%d %H:%M' , errors='coerce') if not pd.isnull(x) else x)
    to_drop = ['Érk.', "Ind."]
    train_data.drop(columns=to_drop, inplace=True)
    train_data['VONAT'] = train_no
    train_data['KOZLEKEDESI_NAP'] = date
    train_data['NEV'] = train_name
    train_data['TiPUS'] = train_type
    train_data['PLUSZ'] = info
    return train_data


def process_plan_fact_time_cols(erk, ind):
    # NaN-nál lehaé
    erk_teny = np.NaN
    erk_terv = np.NaN
    if not pd.isnull(erk):
        erk_a = erk.split()

        erk_terv = erk_a[0]
        if len(erk_a) == 2:
            erk_teny = erk_a[1]
    ind_teny = np.NaN
    ind_terv = np.NaN
    if not pd.isnull(ind):
        ind_a = ind.split()
        ind_terv = ind_a[0]
        if len(ind_a) == 2:
            ind_teny = ind_a[1]
    return erk_teny, erk_terv, ind_teny, ind_terv


def process_train_desc(erk, ind, input):
    new_cols_station = ['KOZLEKEDESI_NAP', 'ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY', 'VONATSZAM', 'VONATTIPUS',
                        'IND_VEGALLOMASROL', 'VEGALLOMAS', 'ERK_VEGALLOMASRA']

    d = datetime.date.today()
    input = input.replace(u'\xa0', " ")
    input = " ".join(input.split())
    ia = input.split(" ")
    arrival_time = ia[-1]
    train_no = ia[0]
    i = 1
    train_type = ""
    other_station = ""
    while (not "--" in ia[i]) and (not ":" in ia[i]):
        train_type = ' '.join([train_type, ia[i]])
        i += 1
    start_time = ia[i]
    i = -2
    while (not "--" in ia[i]) and (not ":" in ia[i]):
        other_station = ' '.join([ia[i], other_station])
        i -= 1
    erk_teny, erk_terv, ind_teny, ind_terv = process_plan_fact_time_cols(erk, ind)
    return pd.Series(dict(zip(new_cols_station,
                              [d, erk_terv, erk_teny, ind_terv, ind_teny, train_no, train_type, start_time,
                               other_station, arrival_time])))


# process_train_desc(ind,erk,inp)

def get_station_data(name):
    print(f'Getting data for {name}')
    station_resp = pull_station_data_from_API(name)
    station_dict = json.loads(station_resp.text)
    stat_df = pd.read_html(StringIO(station_dict['d']['result']))[0]
    new_col_dict = {c:c[2] for c in stat_df.columns}
    stat_df.columns= new_col_dict.values()
    station_data=pd.concat([stat_df, stat_df.apply(lambda x: process_train_desc(x["Érk."],x["Ind."],x["Vonat  Viszonylat"]),axis=1)], axis=1)
    station_data[['ERK_TERV','ERK_TENY','IND_TERV','IND_TENY','IND_VEGALLOMASROL','ERK_VEGALLOMASRA']]=station_data[['ERK_TERV','ERK_TENY','IND_TERV','IND_TENY','IND_VEGALLOMASROL','ERK_VEGALLOMASRA']].apply(lambda x: pd.to_datetime(x, format='%H:%M',errors='coerce').dt.time)
    to_drop=['Érk.',"Ind.","Vonat  Viszonylat"]
    station_data.drop(columns=to_drop,inplace=True)
    station_data['ALLOMAS']=name
    #kozlekedesi nap javítása TODO gondoljuk át
    station_data.loc[station_data['ERK_TERV']<station_data['IND_VEGALLOMASROL'],'KOZLEKEDESI_NAP']=station_data['KOZLEKEDESI_NAP']-datetime.timedelta(days=1)
    return station_data

# minden figyelt volnalra, állomásonként lehívjuk a menetrendet, a jelölt vonataink azok lesznek, amik legalább 3 állomáson átmennek
# ezt naponta, mondjuk 00:01-kor lwhúzzuk, hogy tudjuk mi a terv aznapra
def get_trains_on_lines(main_stations):
    trains_per_lines = {}
    for k, v in main_stations.items():
        print(k, v)
        stat_df_s = {}
        for station in v:
            stat_df_s[station] = get_station_data(station)
        station_df = pd.concat(stat_df_s.values(), axis=0)
        vc = station_df['VONATSZAM'].value_counts()
        #melyik vonatok mennek át legalább 3 állomáson
        trains_per_lines[k] = vc[vc >= 3].index.to_list()
        #upsert('station_events',station_df,engine)
    return trains_per_lines

def load_location_data():
    return pd.read_pickle(generated_files_path+"stat_coord_dict.pkl")

def get_daily_weather_forcast(lat=47.969911,lon=21.767344):
    """
       Loads locations of used meteorlogical stations
       Returns:
       pd.Dataframe: name, lat, lon
    """
    url=f"https://api.openweathermap.org/data/2.5/forecast?units=metric&lat={lat}&lon={lon}&appid={owm_key}"
    response = requests.get(url)
    return process_forecast(response.json())

def process_forecast(wf_dict):
    print(wf_dict)
    """
       Takes a dict received from OWM, and calculates daily weather forcast
       Returns:
       Dict:
        tx : max temp,
        t: avg temp,
        tn: min temo
        r: precipitation
    """
    txa=[]
    tna=[]
    pa=[]
    for rc in wf_dict['list'][:24]:
        txa.append(float(rc['main']['temp_max']))
        tna.append(float(rc['main']['temp_min']))
        p=0.0
        if float(rc['pop'])>0.0:
            if 'rain' in rc:
                p += float(rc['pop']) * float(rc['rain']['3h'])
            if 'snow' in rc:
                p += float(rc['pop']) * float(rc['snow']['3h'])
        pa.append(p)
        tx=max(txa)
        tn=min(tna)
        p=sum(pa)
    return {'tx':tx,'t':(tx+tn)/2,'tn':tn,'r':p}

#def load_model():
def pull_recents_trains_data_from_API():
    train_query={
      "a": "TRAINS",
      "jo": {
        "history": True,
        "id": True
      }
    }
    response = requests.post(mav_api_url, json=train_query)
    return response


# kelleni fog: 'MENETREND_IDO (m)','ELOZO_SZAKASZ_KESES (m)','KESES (m)','tx','t','tn','r','TERV_IDOTARTAM (m)'
def convert_real_time_to_ml_data(input_df, weather_series):
    date_format = '%Y-%m-%d %H:%M:%S'

    # input_df['ERK_TERV'].fillna(input_df['IND_TERV'], inplace=True)
    # input_df['IND_TERV'].fillna(input_df['ERK_TERV'], inplace=True)

    # először szétválasztjuk az eseményeket
    departures = input_df[['Állomás', 'IND_TERV', 'IND_TENY', 'KOZLEKEDESI_NAP']]
    departures.rename(columns={'IND_TERV': 'IDO', 'IND_TENY': 'TENY_IDO'}, inplace=True)

    arrivals = input_df[['Állomás', 'ERK_TERV', 'ERK_TENY', 'KOZLEKEDESI_NAP']]

    arrivals.rename(columns={'ERK_TERV': 'IDO', 'ERK_TENY': 'TENY_IDO'}, inplace=True)
    ml_df = pd.concat([arrivals, departures]).sort_values(by='IDO')
    ml_df = ml_df.dropna(axis=0, subset=['IDO'], thresh=1)

    ml_df['OSSZ_KESES'] = ml_df['TENY_IDO'] - ml_df['IDO']
    ml_df[['ELOZO_OSSZ_KESES', 'ELOZO_ALLOMAS', 'ELOZO_ESEMENY_IDO']] = ml_df[['OSSZ_KESES', 'Állomás', 'IDO']].shift()
    # ahol nincsen tényidő és nincs előrejelzés, ott kell számolnunk.
    ml_df['ELOZO_ESEMENY_IDO'].fillna(ml_df['IDO'], inplace=True)

    ml_df['KESES'] = ml_df['OSSZ_KESES'] - ml_df['ELOZO_OSSZ_KESES']
    ml_df['KESES'].fillna(ml_df['OSSZ_KESES'], inplace=True)

    ml_df['OSSZ_KESES (m)'] = ml_df['OSSZ_KESES'] / np.timedelta64(1, 'm')
    ml_df['TERV_IDOTARTAM (m)'] = (ml_df['IDO'] - ml_df['ELOZO_ESEMENY_IDO']).dt.seconds / 60
    ml_df['KESES (m)'] = ml_df['KESES'] / np.timedelta64(1, 'm')
    ml_df['MENETREND_IDO (m)'] = ml_df['IDO'].dt.hour * 60 + ml_df['IDO'].dt.minute

    ml_df.drop(columns=['KESES', 'OSSZ_KESES', 'ELOZO_ESEMENY_IDO'])

    ml_df['TENY_IDOTARTAM (m)'] = ml_df['TERV_IDOTARTAM (m)'] + ml_df['KESES (m)']
    ml_df[['ELOZO_SZAKASZ_TERV_IDOTARTAM (m)', 'ELOZO_SZAKASZ_KESES (m)']] = ml_df[
        ['TERV_IDOTARTAM (m)', 'KESES (m)']].shift()

    series_repeated = pd.concat([weather_series] * len(ml_df), axis=1).T.reset_index(drop=True)
    print(series_repeated)
    series_repeated.index = ml_df.index
    # Append the series to the DataFrame
    ml_df = pd.concat([ml_df, series_repeated], axis=1)

    return ml_df

def iterative_prediction(iterative_pred_df,model):
    last_pred=None
    preds=[]
    #iterative_pred_df['pred']=np.NaN
    for i in iterative_pred_df.index:
        if last_pred:
            iterative_pred_df.at[i,'ELOZO_SZAKASZ_KESES (m)']=last_pred
        r = iterative_pred_df.loc[[i]]
        #print('last_p', last_pred)
        #rdf=pd.DataFrame(r).T
        #print(rdf)
        p=model.predict(r)
        #print(p)
        preds.append(p[0])
        last_pred=p[0]
    iterative_pred_df['pred']=preds
import pickle
from config import gtfs_dowload_location
def load_geom_dbs():

    file = open(generated_files_path + 'gtfs_shapes_full.pkl', 'rb')
    mapping = pickle.load(file)
    # több shape_id egy rout id-hoz. meg kellene találni a leghosszabbat a az egész vonalhoz..
    shapes = pd.read_csv(gtfs_dowload_location + "latest/gtfs/shapes.txt")
    mapping_with_shapes = mapping.merge(shapes, how='left')
    agg_method='union_of_shapes'
    agg_method='biggest_shape'
    if agg_method=='biggest_shape':
        a = mapping_with_shapes.groupby(['VONATSZAM', 'shape_id'])['shape_pt_sequence'].nunique().reset_index()
        max_indices = a.groupby('VONATSZAM')['shape_pt_sequence'].idxmax()
        max_values = a.loc[max_indices]
        #single_shape_ids_with_train_ids = max_values.merge(mapping, how='left')
        train_desc_with_geom_ids = max_values.merge(shapes, how='left')
        train_desc_with_geom_ids['shape_id'] = train_desc_with_geom_ids['shape_id'].astype(int)
        train_desc_with_geom_ids = train_desc_with_geom_ids[['VONATSZAM', 'shape_id']]
        train_with_shapes = train_desc_with_geom_ids.merge(shapes, how='left')
    else:
        train_with_shapes = mapping_with_shapes.merge(shapes, how='left')




    mapping_with_shapes = train_with_shapes[
        ['VONATSZAM', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence']].drop_duplicates().sort_values(by='shape_pt_sequence')

    return mapping_with_shapes

import geojson
def get_geometry(train_no, mapping_df):
    geom_act = mapping_df[mapping_df['VONATSZAM'] == train_no]
    print(f' {train_no} geometry:')
    print(geom_act)

    if geom_act.empty():
        print(f'No geometry for {train_no}')
        return None
    print(geom_act)
    points = list(zip(geom_act['shape_pt_lon'], geom_act['shape_pt_lat']))
    # Define coordinates for the LineString
    coordinates = points

    # Create a LineString object
    line = geojson.LineString(coordinates)

    # Convert the LineString object to a JSON string
    # line_json = geojson.dumps(line)
    return line



def transform_recent_trains(recent_trains):
    recent_trains=recent_trains[['@Delay','@TrainNumber','@Lat','@Lon','VONAT_','NEV_first','@Relation','KOZLEKEDESI_NAP_last']]
    recent_trains['Vonatszam']=recent_trains['VONAT_'].astype(int)
    recent_trains['Nev']=recent_trains['NEV_first']+' : '+recent_trains['@Relation']
    recent_trains['Nap']=recent_trains['KOZLEKEDESI_NAP_last'].astype(str)
    recent_trains=recent_trains.rename(columns={'@Lat':'Lat','@Lon':'Lon','@Delay':'Keses'})
    return recent_trains[['Vonatszam','Lat','Lon','Nev','Nap','Keses']]
import json
def get_recent_trains(train_schedules):
    resp = pull_recents_trains_data_from_API().json()
    ts = resp['d']['result']['@CreationTime']
    print(ts)
    trains_json = resp['d']['result']['Trains']['Train']
    current_locations = pd.DataFrame(trains_json)
    current_trains=current_locations.merge(train_schedules, right_on="VONATSZAM_L",left_on="@TrainNumber",how='inner')
    current_trains=transform_recent_trains(current_trains)
    resp={"Timestamp":ts,"trains":current_trains.to_dict('records')}
    return current_trains,json.dumps(resp)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        '''
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        '''
        if isinstance(obj, datetime.time):
            return str(obj)
        if isinstance(obj, datetime.date):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return str(obj)
        if isinstance(obj,geojson.LineString):
            return geojson.dumps(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(pd.to_datetime(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from config import main_stations,collected_trains,collected_trains,generated_files_path,weather_folder
import pandas as pd


def get_geometry(train_no, mapping_df):
    geom_act = mapping_df[mapping_df['VONATSZAM'] == train_no]
    points = list(zip(geom_act['shape_pt_lon'], geom_act['shape_pt_lat']))
    # Define coordinates for the LineString
    coordinates = points

    # Create a LineString object
    line = geojson.LineString(coordinates)

    # Convert the LineString object to a JSON string
    # line_json = geojson.dumps(line)
    return line


def process_api_trains(rets):
    routes = pd.read_csv(gtfs_dowload_location + "latest/gtfs/routes.txt")
    routes['numbers'] = routes["route_long_name"].map(
        lambda long_name: ' '.join([i for i in str(long_name).strip().split(" ") if i.isnumeric()])
    )
    # számok és ic nevek
    routes['name'] = ""
    routes.loc[~routes['route_long_name'].isnull(), 'name'] = routes['route_long_name']
    routes.loc[~routes['route_short_name'].isnull(), 'name'] = routes['route_short_name']
    routes = routes[['route_id','numbers','name']].drop_duplicates()
    trips = pd.read_csv(gtfs_dowload_location + "latest/gtfs/trips.txt")
    #trips = trips[trips['shape_id'].notna()]
    trips.loc[trips['shape_id'].isna(), 'shape_id'] = trips['trip_id'].apply(lambda x: float(x.split('_')[0]))

    trips = trips[['route_id', 'shape_id']].drop_duplicates()
    shapes = routes.merge(trips)
    from config import not_existing

    naming_data_from_api = {}
    for k, v in rets.items():
        print(k)
        #feleslegesnek tűnik
        #if k in not_existing:
            #print('not existing ', k)
            #continue
        #ide se nagyon jövünk be
        if v is None:
            print(f'missing train: {k} ')
            not_existing += [k]
            continue
        #API-ból visszafejtettnevek és vonatszámok
        naming_data_from_api[k] = {}
        name = v['NEV'].loc[0]
        type = v['TiPUS'].loc[0]
        if type.isupper():
            print(type)
            t = name
            name = type
            type = t
        if name.islower():
            t = name
            name = type
            type = t
        naming_data_from_api[k]['NEV'] = name
        naming_data_from_api[k]['TIPUS'] = type
        naming_data_from_api[k]['VONATSZAM'] = k
    mapping = pd.DataFrame(naming_data_from_api.values())
    mapping.loc[mapping['NEV'] == "", 'NEV'] = mapping['VONATSZAM'].astype(str)
    #full join:  mindent mindennel, hogy kb név összehasonlítást tudjunk csinálni
    '''
    merged_df = mapping.assign(key=1).merge(shapes.assign(key=1), on='key')
    merged_df['contains'] = merged_df.apply(lambda row: row['NEV'] in row['name'], axis=1)
    # csak azokat tartjuk meg ahol volt név match
    merged_df = merged_df[merged_df['contains']]
    merged_df = merged_df[merged_df['NEV'] != '']
    merged_df.tail()
    merged_df_all_route_ids=merged_df.drop_duplicates()
    merged_df_all_route_ids.to_pickle(generated_files_path + 'gtfs_shapes_full.pkl')
    del merged_df, merged_df_all_route_ids, mapping, trips, routes
    '''
    shapes_with_nums = shapes.loc[shapes['numbers'] != ""]
    shapes_with_names = shapes.loc[shapes['numbers'] == ""]
    shapes_with_nums['VONATSZAM'] = shapes_with_nums['numbers'].astype(int)
    m1 = mapping.merge(shapes_with_nums)
    m1 = m1[['NEV', 'VONATSZAM', 'route_id', 'shape_id']].drop_duplicates()
    shapes_with_names['NEV'] = shapes_with_names['name']

    shapes_with_names = shapes_with_names[['route_id', 'NEV', 'shape_id']].drop_duplicates()

    m2 = mapping.merge(shapes_with_names)
    m2 = m2[['NEV', 'VONATSZAM', 'route_id', 'shape_id']].drop_duplicates()

    m = pd.concat([m1, m2])
    m.to_pickle(generated_files_path + 'gtfs_shapes_full.pkl')

    del m1, m2, shapes_with_names, trips, routes, shapes_with_nums


def init_schedule(main_stations_dict,collected_trains_dict):
    # mi az amit követünk, illetve mi az ami tényleg közlekedik a vonalon
    trains_to_watch = get_trains_on_lines(main_stations_dict)
    # az aznapi menetrend szerint közlekedő vonatok és a historikus adatokban szereplők metszete
    for k,v in collected_trains_dict.items():
        trains_to_watch[k] = list(set(map(int, trains_to_watch[k])).intersection(collected_trains_dict[k]))
    # figyelt vonatok végleges listája
    watched_trains_list=[]
    for k,v in trains_to_watch.items():
        watched_trains_list+=v
    all_watched_train_data={}
    for train_no in watched_trains_list:
        all_watched_train_data[train_no]=get_train_data(train_no)
    process_api_trains(all_watched_train_data)
    #pickle.dump(all_watched_train_data, open(generated_files_path + 'full_train_pull.pkl', 'wb'))
    train_schedules=pd.concat( [v.groupby('VONAT').agg({'IND_TERV': ['first'], 'ERK_TERV': ['last'], 'KOZLEKEDESI_NAP': ['last'],'Állomás':['first','last'],'NEV':'first'}).reset_index() for v in all_watched_train_data.values()])
    train_schedules.columns=train_schedules.columns.map('_'.join)
    train_schedules["VONATSZAM_L"]=train_schedules['VONAT_'].astype(str).apply(lambda x:'55'+x)
    return train_schedules

def add_weather_data(train_schedules,coords,met_stat_locations):
    train_schedules=train_schedules.merge(coords,how='left', left_on=('Állomás_first'),right_on='POLGARI_NEV')
    train_schedules=train_schedules.merge(met_stat_locations,left_on='Legközelebbi met. állomás',right_on='Loc')
    query_data=train_schedules[['Loc','Lat','Lon']].drop_duplicates()
    query_data[['tx','t','tn','r']]=query_data.apply(lambda x: pd.Series(get_daily_weather_forcast(x['Lat'],x['Lon'])),axis=1)
    train_schedules=train_schedules.merge(query_data)
    return train_schedules

def init_data_offline():
    coords=pd.read_pickle(generated_files_path+"stat_coord_dict.pkl")
    weather_meta_file_name=weather_folder+"weather_meta_avg.csv"
    met_stat_locations=pd.read_csv(weather_meta_file_name,sep=',',encoding='iso-8859-2')
    return None,met_stat_locations,coords

def update_gtfs():

    download_gtfs_zip()
    import zipfile
    from config import gtfs_dowload_location

    extracted_folder = gtfs_dowload_location + "latest/gtfs"

    print(f'Extracting {gtfs_dowload_location + "latest/gtfs"} ')
    # create folder with the same name of zip, and extract content in it
    with zipfile.ZipFile(gtfs_dowload_location + "latest/gtfs.zip", 'r') as zip_ref:
        if not os.path.exists(extracted_folder):
            os.mkdir(extracted_folder)
        zip_ref.extractall(extracted_folder)
        # iterate over all the downloaded files (per zip, with hourly update we expect a single file)
        for f in os.listdir(extracted_folder):
            print(f)

def init_data():
    update_gtfs()
    coords=pd.read_pickle(generated_files_path+"stat_coord_dict.pkl")
    weather_meta_file_name=weather_folder+"weather_meta_avg.csv"
    met_stat_locations=pd.read_csv(weather_meta_file_name,sep=',',encoding='iso-8859-2')

    train_schedules=init_schedule(main_stations_dict=main_stations,collected_trains_dict=collected_trains)
    train_schedules=add_weather_data(train_schedules,coords,met_stat_locations)
    return train_schedules,met_stat_locations,coords

def transform_recent_trains(recent_trains):
    recent_trains=recent_trains[['@Delay','@TrainNumber','@Lat','@Lon','VONAT_','NEV_first','@Relation','KOZLEKEDESI_NAP_last']]
    recent_trains['Vonatszam']=recent_trains['VONAT_'].astype(int)
    recent_trains['Nev']=recent_trains['NEV_first']+' : '+recent_trains['@Relation']
    recent_trains['Nap']=recent_trains['KOZLEKEDESI_NAP_last'].astype(str)
    recent_trains=recent_trains.rename(columns={'@Lat':'Lat','@Lon':'Lon','@Delay':'Keses'})
    return recent_trains[['Vonatszam','Lat','Lon','Nev','Nap','Keses']]
import json
def get_recent_trains(train_schedules):
    resp = pull_recents_trains_data_from_API().json()
    ts = resp['d']['result']['@CreationTime']
    #print(ts)
    trains_json = resp['d']['result']['Trains']['Train']
    current_locations = pd.DataFrame(trains_json)
    current_trains=current_locations.merge(train_schedules, right_on="VONATSZAM_L",left_on="@TrainNumber",how='inner')
    current_trains=transform_recent_trains(current_trains)
    resp={"Timestamp":ts,"trains":current_trains.to_dict('records')}
    return current_trains,json.dumps(resp)
def create_train_obj(train_no,lat, lon,name,day,  data_df,delay,type,plus_info='',mapping_with_shapes=None, schedule=False,geom=False):
    train={}
    train['Vonatszam']=train_no
    train['Lat']=lat
    train['Lon']=lon
    train['Nev']=name
    train['Nap']=day
    train['Keses']=delay
    train['TiPUS'] =type
    train['PLUSZ']=plus_info
    if schedule:
        df_copy=data_df.copy()
        df_copy = df_copy.fillna(np.nan).replace([np.nan], [None])
        train['Table']=df_copy.to_dict(orient='records')
    if geom:
        pattern = r"^55"
        replacement = ""
        train_no_1 = int(re.sub(pattern, replacement, str(train_no)))
        train['geom']=get_geometry(train_no_1,mapping_with_shapes)
    return train


def get_recent_train_details(no, train_schedules, model, recent_trains,mapping_with_shapes):
    short_no = no
    if not str(no).startswith('55'):
        short_no = no
        no = int('55' + str(no))
    else:
        short_no = int(str(no).replace('55', ''))
    data = get_train_data(no)
    # nem kellene crashelnie
    sch = train_schedules[train_schedules['VONATSZAM_L'].astype(int) == no].iloc[0]
    weather = sch[['tx', 't', 'tn', 'r']]
    # töröljük az előjelzést

    data[['ERK_TENY', 'IND_TENY']] = data[['ERK_TENY', 'IND_TENY']].map(
        lambda x: pd.NaT if x > datetime.datetime.now() else x)
    data_sv = data[['Km', 'Állomás', 'TiPUS', 'NEV']]
    ml_data = convert_real_time_to_ml_data(data, weather)
    # csak az indulásnál lesz elvileg -TODO randa
    ml_data.loc[0, 'ELOZO_SZAKASZ_KESES (m)'] = 0.0
    td = ml_data.copy()
    past_data = td[~pd.isnull(td['KESES (m)'])]
    print('past_data')
    print(past_data)
    if not past_data.empty:
        past_data['pred'] = model.predict(
            past_data[['MENETREND_IDO (m)', 'ELOZO_SZAKASZ_KESES (m)', 'tx', 't', 'tn', 'r', 'TERV_IDOTARTAM (m)']])
        past_data['Last prediction'] = past_data['IDO'] + pd.to_timedelta(past_data['pred'], unit='m')
        past_data = past_data.drop(columns=['pred'])

    future_data = td[pd.isnull(td['KESES (m)'])]
    #print('Future - ', future_data.shape)
    print('future_data')
    print(future_data)
    iterative_pred_df = future_data[
        ['MENETREND_IDO (m)', 'ELOZO_SZAKASZ_KESES (m)', 'tx', 't', 'tn', 'r', 'TERV_IDOTARTAM (m)']]
    iterative_prediction(iterative_pred_df, model)
    future_data[['ELOZO_SZAKASZ_KESES (m)', 'pred']] = iterative_pred_df[['ELOZO_SZAKASZ_KESES (m)', 'pred']]
    future_data['cum_pred'] = future_data['pred'].cumsum()
    future_data['Last prediction'] = future_data['IDO'] + pd.to_timedelta(future_data['cum_pred'], unit='m')
    future_data = future_data.drop(columns=['pred', 'cum_pred'])
    predicted_df = pd.concat([past_data, future_data])
    print(predicted_df.columns)
    ret_df = predicted_df.groupby('Állomás').agg(
        ERK_TERV=('IDO', 'first'),
        ERK_TENY=('TENY_IDO', 'first'),
        IND_TERV=('IDO', 'last'),
        IND_TENY=('TENY_IDO', 'last'),
        Utolso_erk_pred=('Last prediction', 'first'),
        Utolso_ind_pred=('Last prediction', 'last')
    ).reset_index()
    final = ret_df.sort_values(by='ERK_TERV', ascending=True).merge(data_sv)
    #print(short_no)
    #print(recent_trains['Vonatszam'].unique())
    ans = recent_trains.query(f' Vonatszam == {short_no}' ).head(1)
    if not ans.empty:
        lat=ans['Lat'].values[0]
        lon=ans['Lon'].values[0]
        delay = ans['Keses'].values[0]
        day=ans['Nap'].values[0]
        name=ans['Nev'].values[0]
    else:
        lat = None
        lon = None
        delay = None
        day = None
        name = None
    tipus=final['TiPUS'].values[0]
    resp = {'Timestamp': str(datetime.datetime.now()), 'trains': []}
    resp['trains'].append(create_train_obj(train_no=no, lat=lat, lon=lon, name=name, day=day, data_df=final, delay=delay,type=tipus,
                                      mapping_with_shapes=mapping_with_shapes, schedule=True, geom=True))

    return json.dumps(resp, cls=JSONEncoder)



