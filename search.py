import numpy as np
from mpi4py import MPI
import json
import time
import argparse

# parse the command line arguments
parser = argparse.ArgumentParser(description="multicultural of sydney")
parser.add_argument('--twitter_path', type=str, help='file path for twitter')
parser.add_argument('--grid_path', type=str, help='file path for grid')
args = parser.parse_args()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# record starting time
start_time = time.time()

# Grid file and Twitter file
GRID_FILE_PATH = args.grid_path
TWITTER_FILE_PATH = args.twitter_path

# Read grid file to get 16 initial location grids
def construct_location_grids(file_name):
    loc_grids = {}

    with open(file_name) as f:
        data = json.load(f)
        for loc in data['features']:
            grids = {}
            grids['id'] = loc['properties']['id']
            arr = np.asarray(loc['geometry']['coordinates'][0])
            grids['xmin'] = np.min(arr[:,0])
            grids['xmax'] = np.max(arr[:,0])
            grids['ymin'] = np.min(arr[:,1])
            grids['ymax'] = np.max(arr[:,1])
            grids['count'] = 0
            grids['lang'] = {}
            # loc_grids.append(grids)
            loc_grids[loc['properties']['id']] = grids
    
    return loc_grids

# find corresponding grid for the coords
# the coords has id, longitude, latitude, language of one twitter
def match_grids(loc_grids, coords):  
    lat = coords['lat']
    lon = coords['lon']
    lang = coords['lang']
    
    for grid_id in loc_grids.keys():
        grid = loc_grids[grid_id]
        if lon > grid['xmin'] and lon <= grid['xmax'] and lat > grid['ymin'] and lat <= grid['ymax']:
            # if match, update count and language for that grid
            grid['count'] = grid['count'] + 1
            if lang in grid['lang'].keys():
                grid['lang'][lang] = grid['lang'][lang] + 1
            else:
                grid['lang'][lang] = 1

# sort the language for each grid from high to low
def sort_language(combined_grid):
    for id in combined_grid.keys():
        if combined_grid[id]['count'] != 0:
            dic = dict(sorted(combined_grid[id]['lang'].items(), reverse=True, key=lambda item: item[1]))
            combined_grid[id]['lang'] = dic

# display the final result
def print_result(combined_grid):
    # display the result
    print('cell     #Total Tweets     #Number of Languages used     #Top 10 Languages & #Tweets')
    for id in combined_grid.keys():
        total_tweets = combined_grid[id]['count']
        num_of_lang = len(combined_grid[id]['lang'])
        top_10_lang = ()
        if num_of_lang > 10:
            top_10_lang = tuple(combined_grid[id]['lang'])[0:10]
        else:
            top_10_lang = tuple(combined_grid[id]['lang'])
        print(str(id) + '          ' + str(total_tweets) + '                    ' + str(num_of_lang) + '                         ' + str(top_10_lang))


# Let process 0 count the total number of lines
total_line = 0
if rank == 0:
    with open(TWITTER_FILE_PATH) as f:
        for line in f:
            total_line = total_line + 1
    total_line = total_line - 1
    if size >= 2:
        for i in range(0, size):
            comm.send(total_line, dest=i)
else:
    total_line = comm.recv(source=0)  

# calculate the chunk size for own process
# e.g. total_line = 1000, if running 4 cores, process 0 only need to read 1-250, process 1 need to read 251-500
chunk_size = total_line // size
start_line = chunk_size * rank + 1
end_line = chunk_size * (rank + 1) if rank < size - 1 else total_line

# start to read that chunk of the json file
valid_coords = []
with open(TWITTER_FILE_PATH) as f:
    line_count = 0
    for line in f:
        line_count = line_count + 1
        if line_count == 1:
            continue
        if line_count >= start_line and line_count <= end_line:
            coords = {}
            if line_count < total_line:
                line = line[0:len(line)-2] # need to crop the ending ','
            data = json.loads(line)
            if data['doc']['coordinates'] != None:
                coords['id'] = data['id']
                coords['lat'] = data['doc']['coordinates']['coordinates'][1]
                coords['lon'] = data['doc']['coordinates']['coordinates'][0]
                coords['lang'] = data['doc']['lang']
                valid_coords.append(coords)
        if line_count > end_line:
            break

# match the grids
loc_grids = construct_location_grids(GRID_FILE_PATH)
for data in valid_coords:
     match_grids(loc_grids, data)

if size >= 2:
    loc_grids_all_processes = comm.gather(loc_grids, root = 0)

    if rank == 0:
        combined_grid = construct_location_grids(GRID_FILE_PATH)
        for process_loc_grids in loc_grids_all_processes:
            for id in process_loc_grids.keys():
                if process_loc_grids[id]['count'] == 0:
                    continue
                else:
                    # update count and language
                    combined_grid[id]['count'] = combined_grid[id]['count'] + process_loc_grids[id]['count']
                    for lang in process_loc_grids[id]['lang'].keys():
                        if lang in combined_grid[id]['lang'].keys():
                            combined_grid[id]['lang'][lang] = combined_grid[id]['lang'][lang] + process_loc_grids[id]['lang'][lang]
                        else:
                            combined_grid[id]['lang'][lang] = process_loc_grids[id]['lang'][lang]
        print('time running ' + str(size) +  ' process: ' + str(time.time() - start_time))
        sort_language(combined_grid)
        print_result(combined_grid)
else:
    sort_language(loc_grids)
    print('time running 1 process: ' + str(time.time() - start_time))
    print_result(loc_grids)