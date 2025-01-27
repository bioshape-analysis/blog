# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false 
def plot_cell_trajectory(n):
    centroids = centr[n - 1]  
    centroids = np.array(centroids)
    x_coords = centroids[:, 0]  
    y_coords = centroids[:, 1] 

    # riemann_data = riemann[n-1] 
    # riemann_data = np.array(riemann_data)  

    time_steps = np.arange(len(x_coords))  

    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_coords[0], y_coords[0],  
        c='black',
        marker='o',
        edgecolor='k',
        s=100,
        alpha=0.7,
        label='Start Point'
    )
    scatter = plt.scatter(
        x_coords[1:], y_coords[1:],  
        c=time_steps[1:],  #riemann_data[1:],        
        cmap='plasma',             
        marker='o',
        edgecolor='k',
        s=100,
        alpha=0.7
    )
    plt.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)  

    plt.title(f"Cell Num {n}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Step (t)', rotation=270, labelpad=15)

    plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false
loaded = load("../trajectory_data.mat");
allTracks = loaded.tracks;

probDim = 2;       
plotRes = 0;       
peakAlpha = 95;  

results = struct();

trackNames = fieldnames(allTracks);
for i = 1:length(trackNames)
    trackName = trackNames{i};
    tracks = allTracks.(trackName);

    [transDiffResults, errFlag] = basicTransientDiffusionAnalysisv1(tracks, probDim, plotRes, peakAlpha);
    
    if isfield(transDiffResults.segmentClass, 'momentScalingSpectrum')
        results.(trackName).momentScalingSpectrum = transDiffResults.segmentClass.momentScalingSpectrum;
    end
end

h5FileName = 'time_events.h5';

trackNames = fieldnames(results);
for i = 1:length(trackNames)
    trackName = trackNames{i};
    data = results.(trackName).momentScalingSpectrum;
end
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false

def get_motion_type_distribution():
    type_counts = defaultdict(int)
    total_count = 0

    with h5py.File('time_events_95.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            interval_types = first_three_rows[2, :]
            
            for interval_type in interval_types:
                if not np.isnan(interval_type):
                    type_counts[int(interval_type)] += 1
                    total_count += 1
                else:
                    type_counts["unclassified"] += 1
                    total_count += 1

    proportions = {key: count / total_count for key, count in type_counts.items()}
    return proportions

def plot_motion_type_distribution():

    proportions = get_motion_type_distribution()


    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    colors = ["brown", "blue", "cyan", "magenta", "black"]
    keys = [0, 1, 2, 3, "unclassified"]

    values = [proportions.get(key, 0) for key in keys]
    plt.figure(figsize=(8, 6))
    plt.pie(
        values,
        labels=types,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={'color': 'white', 'fontsize': 12}  
    )
    plt.title("Proportion of Motion Types")
    plt.tight_layout()
    plt.savefig("motion_type_distribution_90.png")
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false

def compute_transition_matrix_with_unclassified():
    transition_counts = np.zeros((5, 5))  
    total_transitions = 0

    with h5py.File('time_events_95.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            interval_types = first_three_rows[2, :]

            for start_idx in range(len(interval_types) - 1):
                type_from = interval_types[start_idx]
                type_to = interval_types[start_idx + 1]


                type_from = int(type_from) if not np.isnan(type_from) else 4  
                type_to = int(type_to) if not np.isnan(type_to) else 4  

                if type_from != type_to:  
                    transition_counts[type_from, type_to] += 1
                    total_transitions += 1


    transition_matrix = (transition_counts / total_transitions) * 100  
    return transition_matrix

def plot_transition_matrix_with_unclassified(transition_matrix):
    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    
    plt.figure(figsize=(8, 6))
    masked_matrix = np.ma.masked_where(np.eye(len(types)), transition_matrix)  
    plt.imshow(masked_matrix, cmap="Blues", aspect="auto")
    for i in range(5):
        plt.fill([i - 0.5, i + 0.5, i + 0.5, i - 0.5],
                 [i - 0.5, i - 0.5, i + 0.5, i + 0.5],
                 color='black')

    plt.colorbar(label="%")

    plt.xticks(ticks=np.arange(len(types)), labels=types, rotation=45)
    plt.yticks(ticks=np.arange(len(types)), labels=types)
    plt.title("Transition Matrix")



    for i in range(len(types)):
        for j in range(len(types)):
            if i != j:  
                plt.text(j, i, f"{transition_matrix[i, j]:.1f}%", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig("transition_matrix_with_unclassified_95.png")
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false
def average_velocity_with_ttest():
    type_data = defaultdict(list)

    with h5py.File('time_events_95.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices) and interval_type in [2, 3]:                      start = event_indices[start_idx] + 1  
                    end = event_indices[start_idx + 1]
                    segment = [get_abs_velocity(plot_index, idx) for idx in range(start, end + 1)]
                    type_data[int(interval_type)].extend(segment)

 
    free_diffusion = type_data[2]
    directed_diffusion = type_data[3]

    t_stat, p_value = ttest_ind(free_diffusion, directed_diffusion, equal_var=False)


    plt.figure(figsize=(8, 6))
    types = ["Free Diffusion", "Directed Diffusion"]
    colors = ["cyan", "magenta"]
    means = []
    conf_intervals = []

    for key, velocities in zip([2, 3], [free_diffusion, directed_diffusion]):
        if velocities:  
            mean = np.mean(velocities)
            ci = sem(velocities) * 1.96  
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.text(0.5, max(means) + max(conf_intervals) * 1.2,
             f"T-test p-value: {p_value:.3e}",
             ha='center', fontsize=12, color='black')

    plt.xticks(x, types)
    plt.ylabel("Velocity")
    plt.tight_layout()
    plt.savefig("mean_velocity_with_ttest_95.png")
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false

def average_angle_with_ttest():
    type_data = defaultdict(list)

    with h5py.File('time_events_90.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices) and interval_type in [2, 3]:                      start = event_indices[start_idx] + 1 
                    end = event_indices[start_idx + 1]
                    segment = [get_velocity_angle_rel(plot_index, idx) for idx in range(start, end + 1)]
                    type_data[int(interval_type)].extend(segment)

    free_diffusion = type_data[2]
    directed_diffusion = type_data[3]

    t_stat, p_value = ttest_ind(free_diffusion, directed_diffusion, equal_var=False)


    plt.figure(figsize=(8, 6))
    types = ["Free Diffusion", "Directed Diffusion"]
    colors = ["cyan", "magenta"]
    means = []
    conf_intervals = []

    for key, angles in zip([2, 3], [free_diffusion, directed_diffusion]):
        if angles: 

            sigma = 2
            angles_smoothed = gaussian_filter1d(angles, sigma=sigma)

            mean = np.mean(angles_smoothed)
            ci = sem(angles_smoothed) * 1.96 
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))


    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.text(0.5, max(means) + max(conf_intervals) * 1.2,
             f"T-test p-value: {p_value:.3e}",
             ha='center', fontsize=12, color='black')

    plt.xticks(x, types)
    plt.ylabel("Angle")
    plt.tight_layout()
    plt.savefig("mean_angle_with_ttest_90.png")
    plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false
riemann_distances = []
a = 1
b = 1/2

CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
    ambient_dim=2, k_sampling_points=1000, equip=False
)
CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)

def calculate_distance(border,reference_shape):

    return CURVES_SPACE_ELASTIC.metric.dist(CURVES_SPACE_ELASTIC.projection(border), CURVES_SPACE_ELASTIC.projection(reference_shape))


for cell_i in range(1, 205):
    number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{cell_i}", entry)) for entry in os.listdir(f"cells/cell_{cell_i}"))  

    iter_distance = np.zeros(number_of_frames)

    BASE_LINE = np.load(f'cells/cell_{cell_i}/frame_1/outline.npy')
    BASE_LINE= interpolate(BASE_LINE,1000)
    BASE_LINE = preprocess(BASE_LINE)
    BASE_LINE= project_on_kendall_space(BASE_LINE)
    for i in range(number_of_frames):
        border_cell = np.load(f'cells/cell_{cell_i}/frame_{i+1}/outline.npy')
        cell_interpolation= interpolate(border_cell,1000)
        cell_preprocess = preprocess(cell_interpolation)
        border_cell = cell_preprocess
        border_cell = project_on_kendall_space(cell_interpolation)
        aligned_border = align(border_cell, BASE_LINE, rescale=True, rotation=False, reparameterization=True, k_sampling_points=1000)
        iter_distance[i] = calculate_distance(aligned_border, BASE_LINE)
        BASE_LINE = aligned_border 

    riemann_distances.append(iter_distance)
### Dividing by delta t in the results.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false
def plot_cell_by_motion_type(n):

    cell_dir = 'cells'
    cell_path = os.path.join(
        cell_dir, 
        sorted(os.listdir(cell_dir), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)[n-1]
    )

    cell = sorted(
        os.listdir(cell_path), 
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
    )
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    interval_colors = {
        0: "brown", 
        1: "blue",       
        2: "cyan",       
        3: "magenta",  
        "unclassified": "black"  
    }
    import h5py

    with h5py.File('time_events_90.h5', 'r') as f:
        track_i_data = f[f'/track_{n}'][:]
        first_three_rows = track_i_data[:3]
        event_indices = first_three_rows[0, :].astype(int) - 1
        interval_types = first_three_rows[2, :]  

    for i, frame in enumerate(cell):
        frame_path = os.path.join(cell_path, frame)
        time = np.load(os.path.join(frame_path, 'time.npy'))
        outline = np.load(os.path.join(frame_path, 'outline.npy'))

        current_type = None
        for start_idx, interval_type in enumerate(interval_types):
            if i >= event_indices[start_idx] and (start_idx + 1 == len(event_indices) or i < event_indices[start_idx + 1]):
                current_type = interval_type
                break

        if current_type is not None:
            interval_type = int(current_type) if not np.isnan(current_type) else "unclassified"
            color = interval_colors.get(interval_type, "black")

        ax.plot3D(
            outline[:, 0], 
            outline[:, 1], 
            np.full(len(outline[:, 1]), time),
            color=color, 
            linewidth=1
        )
        print(f"Frame {frame}: Time = {time}, Motion Type = {current_type}")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="brown", lw=2, label="Immobile"),
        Line2D([0], [0], color="blue", lw=2, label="Confined Diffusion"),
        Line2D([0], [0], color="cyan", lw=2, label="Free Diffusion"),
        Line2D([0], [0], color="magenta", lw=2, label="Directed Diffusion"),
        Line2D([0], [0], color="black", lw=2, label="Unclassified")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time')
    ax.set_title(f"Cell {n}")
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
#| output: false

def average_riemann_distances_with_ttest():
    type_data = defaultdict(list)

    # Собираем данные из файла time_events_95.h5
    with h5py.File('time_events_90.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices) and interval_type in [2, 3]:  # Только Free Diffusion и Directed Diffusion
                    start = event_indices[start_idx] + 1  # Пропускаем i = 0
                    end = event_indices[start_idx + 1]
                    segment = [
                        get_riemann_dist(plot_index, idx) /
                        (get_times(plot_index, idx) - get_times(plot_index, idx - 1))
                        for idx in range(start, end + 1)
                    ]
                    type_data[int(interval_type)].extend(segment)

    # Извлекаем данные для t-теста
    free_diffusion = type_data[2]
    directed_diffusion = type_data[3]

    # Вычисляем t-тест
    t_stat, p_value = ttest_ind(free_diffusion, directed_diffusion, equal_var=False)

    # Построение средних значений Римановых дистанций с доверительными интервалами
    plt.figure(figsize=(8, 6))
    types = ["Free Diffusion", "Directed Diffusion"]
    colors = ["cyan", "magenta"]
    means = []
    conf_intervals = []

    for key, distances in zip([2, 3], [free_diffusion, directed_diffusion]):
        if distances:  # Проверка, что данные есть
            mean = np.mean(distances)
            ci = sem(distances) * 1.96  # 95% доверительный интервал
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    # Рисуем каждую точку отдельно с её цветом и доверительным интервалом
    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    # Отображаем p-value на графике
    plt.text(0.5, max(means) + max(conf_intervals) * 1.2,
             f"T-test p-value: {p_value:.3e}",
             ha='center', fontsize=12, color='black')

    plt.xticks(x, types)
    plt.ylabel("Riemann Velocity")

    plt.tight_layout()
    plt.savefig("riemann_distances_with_ttest_90.png")
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
