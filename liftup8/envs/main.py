import jsbsim
import folium
import random


class Aircraft:
    def __init__(self):
        self.exec = jsbsim.FGFDMExec('./jsbsim/', None)
        self.exec.set_debug_level(0)
        self.exec.load_model("f16")
        self.exec.set_dt(1.0 / 60.)
        self.path = []

    def initialize(self, psi=323):
        self.exec.set_property_value('ic/h-sl-ft', 8.42)
        self.exec.set_property_value('ic/terrain-elevation-ft', 8.42)
        self.exec.set_property_value('ic/h-agl-ft', 8.42)
        self.exec.set_property_value('ic/long-gc-deg', 1.37211666700005708)
        # geocentrique (angle depuis le centre de la terre)
        self.exec.set_property_value('ic/lat-gc-deg', 43.6189638890000424)
        # geodesique
        # self.exec.set_property_value('ic/lat-geod-deg', 43.6189638890000424)
        self.exec.set_property_value('ic/u-fps', 11.8147)
        self.exec.set_property_value('ic/v-fps', 0)
        self.exec.set_property_value('ic/w-fps', 0)
        self.exec.set_property_value('ic/p-rad_sec', 0)
        self.exec.set_property_value('ic/q-rad_sec', 0)
        self.exec.set_property_value('ic/r-rad_sec', 0)
        self.exec.set_property_value('ic/roc-fpm', 0)
        self.exec.set_property_value('ic/psi-true-deg', psi)
        self.exec.run_ic()

    def setGears(self, status='down'):
        if status == 'down':
            self.exec.set_property_value('gear/gear-pos-norm', 1)
            self.exec.set_property_value('gear/unit[1]/pos-norm', 1)
            self.exec.set_property_value('gear/unit[2]/pos-norm', 1)
            # Set gear down
            self.exec.set_property_value('gear/gear-cmd-norm', 1)
        else:
            self.exec.set_property_value('gear/gear-pos-norm', 0)
            self.exec.set_property_value('gear/unit[1]/pos-norm', 0)
            self.exec.set_property_value('gear/unit[2]/pos-norm', 0)
            # Set gear up
            self.exec.set_property_value('gear/gear-cmd-norm', 0)

    def setEngines(self, status='on'):
        propulsion = self.exec.get_propulsion()
        for j in range(propulsion.get_num_engines()):
            propulsion.get_engine(j).init_running()
            propulsion.get_steady_state()
        self.exec.set_property_value('fcs/mixture-cmd-norm', 1)

    def getProperties(self, filter='ic'):
        properties = self.exec.query_property_catalog('')
        filtered = []
        for p in properties:
            if filter in p:
                filtered.append(p)
        return filtered

    def getPath(self):
        return self.path

    def saveFullState(self):
        props = self.exec.get_property_catalog('')
        for p in self.state_trajectory.keys():
            self.state_trajectory[p].append(props[p])

    def step(self, throttle=0.5, steer=0):
        self.exec.set_property_value('fcs/throttle-cmd-norm', throttle)
        self.exec.set_property_value('fcs/steer-cmd-norm', steer)
        for _ in range(5):
            self.exec.run()
        lat = self.exec.get_property_value('position/lat-gc-deg')
        lon = self.exec.get_property_value('position/long-gc-deg')
        speed = self.exec.get_property_value('velocities/vc-fps')
        self.path.append([lat, lon])
        return {'speed': speed}


class Bomber:
    def __init__(self):
        self.index = -1
        self.paths = []
        self.exec = None

    def reset(self):
        # Create the simulation otherwise close it
        if self.exec is not None:
            self.exec = None
            del self.exec  # This one is removing the crash

        self.exec = jsbsim.FGFDMExec('./jsbsim/', None)

        self.exec.set_debug_level(0)
        self.exec.load_model("f16")
        self.exec.set_dt(1.0 / 60.)

        self.index = self.index + 1
        self.paths.append([])

    def initialize(self, psi=323):
        self.exec.set_property_value('ic/h-sl-ft', 8.42)
        self.exec.set_property_value('ic/terrain-elevation-ft', 8.42)
        self.exec.set_property_value('ic/h-agl-ft', 8.42)
        self.exec.set_property_value('ic/long-gc-deg', 1.37211666700005708)
        # geocentrique (angle depuis le centre de la terre)
        self.exec.set_property_value('ic/lat-gc-deg', 43.6189638890000424)
        # geodesique
        # self.exec.set_property_value('ic/lat-geod-deg', 43.6189638890000424)
        self.exec.set_property_value('ic/u-fps', 11.8147)
        self.exec.set_property_value('ic/v-fps', 0)
        self.exec.set_property_value('ic/w-fps', 0)
        self.exec.set_property_value('ic/p-rad_sec', 0)
        self.exec.set_property_value('ic/q-rad_sec', 0)
        self.exec.set_property_value('ic/r-rad_sec', 0)
        self.exec.set_property_value('ic/roc-fpm', 0)
        self.exec.set_property_value('ic/psi-true-deg', psi)
        self.exec.run_ic()

    def setGears(self, status='down'):
        if status == 'down':
            self.exec.set_property_value('gear/gear-pos-norm', 1)
            self.exec.set_property_value('gear/unit[1]/pos-norm', 1)
            self.exec.set_property_value('gear/unit[2]/pos-norm', 1)
            # Set gear down
            self.exec.set_property_value('gear/gear-cmd-norm', 1)
        else:
            self.exec.set_property_value('gear/gear-pos-norm', 0)
            self.exec.set_property_value('gear/unit[1]/pos-norm', 0)
            self.exec.set_property_value('gear/unit[2]/pos-norm', 0)
            # Set gear up
            self.exec.set_property_value('gear/gear-cmd-norm', 0)

    def setEngines(self, status='on'):
        propulsion = self.exec.get_propulsion()

        for j in range(propulsion.get_num_engines()):
            propulsion.get_engine(j).init_running()
            propulsion.get_steady_state()

        self.exec.set_property_value('fcs/mixture-cmd-norm', 1)

    def getProperties(self, filter='ic'):
        properties = self.exec.query_property_catalog('')
        filtered = []
        for p in properties:
            if filter in p:
                filtered.append(p)
        return filtered

    def getPath(self, index):
        return self.paths[index]

    def saveFullState(self):
        props = self.exec.get_property_catalog('')
        for p in self.state_trajectory.keys():
            self.state_trajectory[p].append(props[p])

    def step(self, throttle=0.5, steer=0):
        self.exec.set_property_value('fcs/throttle-cmd-norm', throttle)
        self.exec.set_property_value('fcs/steer-cmd-norm', steer)

        for _ in range(5):
            self.exec.run()

        lat = self.exec.get_property_value('position/lat-gc-deg')
        lon = self.exec.get_property_value('position/long-gc-deg')

        speed = self.exec.get_property_value('velocities/vc-fps')

        self.paths[self.index].append([lat, lon])

        return {'speed': speed}


plane = []

color = [
    'red', 'blue', 'green', 'purple', 'orange', 'white', 'gray', 'black',
    'black'
]

nbiterations = 1000

# Generate steer command between -1 & +1
steering = []
for i in range(nbiterations):
    steering.append(2.0 * random.random() - 1.0)

steering = list(sorted(steering))

speed = []

for i in range(len(color)):
    plane.append(Aircraft())
    plane[i].initialize(psi=i * 36)
    plane[i].setGears('down')
    plane[i].setEngines('on')

    for j in range(nbiterations):
        plane[i].step(throttle=0.5, steer=steering[j])

map = None

map = folium.Map(location=[43.6189638890000424, 1.37211666700005708],
                 zoom_start=15)

folium.Marker((43.6189638890000424, 1.37211666700005708),
              marker_icon='plane').add_to(map)

for i in range(len(color)):
    folium.PolyLine(plane[i].getPath(), color=color[i], weight=2.5,
                    opacity=1).add_to(map)

map

bombers = []

# color = ['red', 'blue', 'green', 'purple', 'orange', 'white', 'gray', 'black', 'black']
color = ['grey']

nbiterations = 1000

# Generate steer command between -1 & +1
steering = []
for i in range(nbiterations):
    steering.append(2.0 * random.random() - 1.0)

steering = list(sorted(steering))

speed = []

bombers.append(Bomber())

bombers[0].reset()
bombers[0].initialize(psi=45)
bombers[0].setGears('down')
bombers[0].setEngines('on')

for j in range(nbiterations):
    bombers[0].step(throttle=0.5, steer=steering[j])

for i in range(10):
    bombers[0].reset()
    bombers[0].initialize(psi=90 + 5 * i)
    bombers[0].setGears('down')
    bombers[0].setEngines('on')

    for j in range(nbiterations):
        bombers[0].step(throttle=0.5, steer=steering[j])

bombers[0].reset()
bombers[0].initialize(psi=45)
bombers[0].setGears('down')
bombers[0].setEngines('on')

for j in range(nbiterations):
    bombers[0].step(throttle=0.5, steer=steering[j])

bmap = folium.Map(location=[43.6189638890000424, 1.37211666700005708],
                  zoom_start=15)

folium.Marker((43.6189638890000424, 1.37211666700005708),
              marker_icon='plane').add_to(bmap)

folium.PolyLine(bombers[0].getPath(0), color='grey', weight=2.5,
                opacity=1).add_to(bmap)

bmap
