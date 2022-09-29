import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulator import *
from MLP import *


def draw_on_pic(line, color):
    xx = np.array([line.d1.x, line.d2.x])
    yy = np.array([line.d1.y, line.d2.y])
    pic_plot.plot(xx, yy, color=color)


def draw_car(color):
    theta = np.arange(0, 2 * np.pi, 0.01)
    xx = car.x + car.r * np.cos(theta)
    yy = car.y + car.r * np.sin(theta)
    pic_plot.plot(xx, yy, color=color)
    xx = np.array([car.x, car.x + car.r *
                   np.cos(to_rad(car.angle))])
    yy = np.array([car.y, car.y + car.r *
                   np.sin(to_rad(car.angle))])
    pic_plot.plot(xx, yy, color=color)


def init():
    pic_plot.clear()
    return load_map()


def init_phase2():
    pic_plot.clear()
    for line in map:
        draw_on_pic(line, 'black')
    for line in end:
        draw_on_pic(line, 'red')
    draw_on_pic(start, 'green')
    draw_car('blue')
    pic_canvas.draw()
    info_label.configure(
        text='Coords: {:.3f}, {:.3f}; Angle: {:.3f}\nFront: {:.3f}, Right: {:.3f}, Left: {:.3f}'.format(car.x, car.y, car.angle, car.distance('front', map),
                                                                                                        car.distance('right', map), car.distance('left', map)))
    window.update()


def update():
    car.run(float(theta_entry.get()), map)
    init_phase2()
    if car.reach(end):  # win
        print('win')
        car.dumplog()
        car.reset(map, start=start)
        init()
        init_phase2()
        return
    if car.reach(map):  # die
        print('die')
        # car.dumplog()
        car.reset(map, start=start)
        init()
        init_phase2()
        return


def browseTrack():
    filename = filedialog.askopenfilename(
        initialdir=".", title="Select a File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))

    # Change label contents
    track_explorer_label.configure(text=filename)


def loadTrack():
    directory = track_explorer_label.cget("text")
    if directory == 'Not chosen yet':
        directory = './軌道座標點.txt'
    load_map(directory=directory)
    track_explorer_label.configure(text='Load complete!')


# def auto():
#     flag = autorun()
#     while not flag:
#         flag = autorun()


# def autorun():
#     fd = car.distance('front', map)
#     ld = car.distance('left', map)
#     rd = car.distance('right', map)
#     if fd > 15:
#         if ld < 6:
#             rnum = random.uniform(0.0, 20.0)
#             car.run(rnum, map)
#         elif rd < 6:
#             rnum = random.uniform(-20.0, 0.0)
#             car.run(rnum, map)
#         else:
#             car.run(0, map)
#     elif fd > 10:
#         rnum = random.uniform(0.0, 20.0)
#         if ld > rd:
#             rnum = -rnum
#         car.run(rnum, map)
#     else:
#         rnum = random.uniform(20.0, 40.0)
#         if ld > rd:
#             rnum = -rnum
#         car.run(rnum, map)
#     init_phase2()
#     if car.reach(end):
#         print('win')
#         car.dumplog()
#         car.reset(map, start)
#         init()
#         init_phase2()
#         return True
#     if car.reach(map):
#         print('die')
#         # car.dumplog()
#         car.reset(map, start)
#         init()
#         init_phase2()
#         return False
#     return False


# def browseData():
#     filename = filedialog.askopenfilename(
#         initialdir="/", title="Select a File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))

#     # Change label contents
#     data_explorer_label.configure(text=filename)


# def train():
#     directory = data_explorer_label.cget("text")
#     if directory == 'Not chosen yet':
#         directory = './output.txt'
#     model.load_data(directory)
#     learnrate = float(learnrate_entry.get())
#     if not learnrate == 0.0:
#         model.learnrate = learnrate
#     max_epoch = int(epoch_entry.get())
#     if not max_epoch == 0:
#         model.max_epoch = max_epoch
#     if model.train() == False:
#         model.load_model()

def browsePath():
    filename = filedialog.askopenfilename(
        initialdir="./output", title="Select a File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))

    # Change label contents
    path_explorer_label.configure(text=filename)


def pathRun():
    directory = path_explorer_label.cget("text")
    if directory == 'Not chosen yet':
        directory = './output/path1.txt'
    with open(directory, 'r') as f:
        line = f.readline()
        tokens = re.findall(r'-?\d+\.*\d*', line)
        car.reset(map, x=float(tokens[0]), y=float(tokens[1]))
        pic_plot.clear()
        init_phase2()
        window.update()
        for line in f:
            tokens = re.findall(r'-?\d+\.*\d*', line)
            print(float(tokens[5]))
            car.run(float(tokens[5]), map)
            pic_plot.clear()
            init_phase2()
            window.update()
    car.reset(map, start=start)
    init_phase2()


def browseModel():
    filename = filedialog.askopenfilename(
        initialdir="./output", title="Select a File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))

    # Change label contents
    model_explorer_label.configure(text=filename)


def load():
    directory = model_explorer_label.cget("text")
    if directory == 'Not chosen yet' or directory == 'Load complete!':
        directory = './output/MLPopt.txt'
    model.load_model(directory)
    model_explorer_label.configure(text='Load complete!')


def modelrun():
    flag = mr()
    while not flag:
        flag = mr()


def mr():
    fd = car.distance('front', map)
    rd = car.distance('right', map)
    ld = car.distance('left', map)

    x = []
    x.append(fd)
    x.append(rd)
    x.append(ld)

    y = model.gety(x)
    car.run(y, map)
    init_phase2()

    window.update()
    if car.reach(end):
        print('win')
        car.dumplog()
        car.reset(map, start=start)
        init()
        init_phase2()
        return True
    if car.reach(map):
        print('die')
        # car.dumplog()
        car.reset(map, start=start)
        init()
        init_phase2()
        return True
    return False


window = tk.Tk()
window.title('AC')
window.geometry('1200x800')
window.configure(background='white')

header_label = tk.Label(window, text='Cars')
header_label.pack()

# 以下為 pic_frame 群組
pic_frame = tk.Frame(window)
pic_frame.pack(side=tk.LEFT)
pic_figure = Figure(figsize=(5, 5))
pic_plot = pic_figure.add_subplot(111)
# pic_plot.set_xlim([0, 50])
# pic_plot.set_ylim([0, 50])
pic_canvas = FigureCanvasTkAgg(
    pic_figure, master=pic_frame)
pic_canvas.get_tk_widget().pack(side=tk.TOP, fill='both', expand=1)
track_label = tk.Label(pic_frame, text="Select track file: ")
track_label.pack(side=tk.LEFT)
track_explorer_label = tk.Label(pic_frame, text='Not chosen yet')
track_explorer_label.pack(side=tk.LEFT)
track_explore_button = tk.Button(
    pic_frame, text="Browse Files", command=browseTrack)
track_explore_button.pack(side=tk.LEFT)
load_track_button = tk.Button(pic_frame, text='Load', command=loadTrack)
load_track_button.pack(side=tk.TOP)

# 以下為 controlls_frame 群組
controlls_frame = tk.Frame(window)
controlls_frame.pack(side=tk.LEFT)
info_label = tk.Label(controlls_frame)
info_label.pack(side=tk.TOP, anchor='w')
theta_frame = tk.Frame(controlls_frame)
theta_frame.pack(side=tk.TOP, anchor='w')
theta_label = tk.Label(theta_frame, text='轉向')
theta_label.pack(side=tk.LEFT)
theta_var = tk.StringVar(value='0.0')
theta_entry = tk.Entry(theta_frame, textvariable=theta_var)
theta_entry.pack(side=tk.LEFT)
run_button = tk.Button(controlls_frame, text='GO',
                       command=update)
run_button.pack(side=tk.TOP)
# randomrun_button = tk.Button(controlls_frame, text='RANDOM GO',
#                              command=auto)
# randomrun_button.pack(side=tk.TOP)
path_label = tk.Label(controlls_frame, text='Select path file:')
path_label.pack(side=tk.TOP, anchor=tk.W)
path_explore_frame = tk.Frame(controlls_frame)
path_explore_frame.pack(side=tk.TOP)
path_explorer_label = tk.Label(path_explore_frame, text='Not chosen yet')
path_explorer_label.pack(side=tk.LEFT)
path_explore_button = tk.Button(
    path_explore_frame, text="Browse Files", command=browsePath)
path_explore_button.pack(side=tk.LEFT)
path_button = tk.Button(
    controlls_frame, text='Reproduce Path', command=pathRun)
path_button.pack(side=tk.TOP)

# 以下為 MLP_frame 群組
MLP_frame = tk.Frame(window)
MLP_frame.pack(side=tk.LEFT, anchor=tk.W)

# data_frame = tk.Frame(MLP_frame)
# data_frame.pack(side=tk.TOP)
# data_label = tk.Label(data_frame, text='Train Data:')
# data_label.pack(side=tk.TOP)
# data_explorer_label = tk.Label(data_frame, text="Not chosen yet")
# data_explorer_label.pack(side=tk.LEFT)
# data_explore_button = tk.Button(
#     data_frame, text="Browse Files", command=browseData)
# data_explore_button.pack(side=tk.LEFT)

# learnrate_frame = tk.Frame(MLP_frame)
# learnrate_frame.pack(side=tk.TOP)
# learnrate_label = tk.Label(learnrate_frame, text='Learn Rate')
# learnrate_label.pack(side=tk.LEFT)
# learnrate_var = tk.StringVar(value='0.0')
# learnrate_entry = tk.Entry(learnrate_frame, textvariable=learnrate_var)
# learnrate_entry.pack(side=tk.LEFT)

# epoch_frame = tk.Frame(MLP_frame)
# epoch_frame.pack(side=tk.TOP)
# epoch_label = tk.Label(epoch_frame, text='Max Epoch')
# epoch_label.pack(side=tk.LEFT)
# epoch_var = tk.StringVar(value='0')
# epoch_entry = tk.Entry(epoch_frame, textvariable=epoch_var)
# epoch_entry.pack(side=tk.LEFT)

# train_button = tk.Button(MLP_frame, text='Train', command=train)
# train_button.pack(side=tk.TOP)

model_label = tk.Label(MLP_frame, text="Select model file:")
model_label.pack(side=tk.TOP, anchor=tk.W)
model_frame = tk.Frame(MLP_frame)
model_frame.pack(side=tk.TOP)
model_explorer_label = tk.Label(model_frame, text="Not chosen yet")
model_explorer_label.pack(side=tk.LEFT)
model_explore_button = tk.Button(
    model_frame, text="Browse Files", command=browseModel)
model_explore_button.pack(side=tk.LEFT)

model_button = tk.Button(MLP_frame, text='Load Model', command=load)
model_button.pack(side=tk.TOP)
modelrun_button = tk.Button(MLP_frame, text='Run With Model', command=modelrun)
modelrun_button.pack(side=tk.TOP)
modelrun_onestep_button = tk.Button(
    MLP_frame, text='Run With Model(1 step)', command=mr)
modelrun_onestep_button.pack(side=tk.TOP)

if __name__ == '__main__':
    model = MLP()
    model.load_model()
    car, end, map, start = init()
    init_phase2()
    window.mainloop()
