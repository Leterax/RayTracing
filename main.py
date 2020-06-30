import math
from pathlib import Path

import moderngl
import pyrr
from pyrr import Matrix44
import numpy as np
from moderngl_window.geometry import quad_fs
import moderngl_window as mglw


def next_power_of_two(x):
    x = int(x)
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x


def normalize(v):
    return v / np.linalg.norm(v)


class RayTracer(mglw.WindowConfig):
    # moderngl_window settings
    gl_version = (4, 3)
    title = "RayTracing demo"
    resource_dir = (Path(__file__) / "../resources").absolute()
    window_size = 1280, 720
    aspect_ratio = None
    tracing_samples = 2
    near_far_planes = (0.0, 2 ** 16)
    mipmap_levels = math.floor(math.log2(tracing_samples))
    resizable = False
    clear_color = 51 / 255, 51 / 255, 51 / 255

    group_size = (32, 32)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.wnd.mouse_exclusivity = True

        self.group_size = next_power_of_two(self.group_size[0]), next_power_of_two(self.group_size[1])
        self.num_work_groups = (
            next_power_of_two(self.window_size[0] * self.tracing_samples) // self.group_size[0],
            next_power_of_two(self.window_size[1] * self.tracing_samples) // self.group_size[1],
        )

        # load programs
        self.program = self.load_compute_shader(
            "compute_shader.glsl",
            defines={
                "GROUP_X": self.group_size[0],
                "GROUP_Y": self.group_size[1],
                "TMIN": self.near_far_planes[0],
                "TMAX": self.near_far_planes[1],
            },
        )
        self.depth_view_program = self.load_program(
            "depth_shader.glsl", defines={"TMIN": self.near_far_planes[0], "TMAX": self.near_far_planes[1]}
        )
        self.view_program = self.load_program("my_shader.glsl")

        self.screen_texture = self.ctx.texture(
            (int(self.window_size[0] * self.tracing_samples), int(self.window_size[1] * self.tracing_samples)),
            components=4,
            dtype="f4",
        )
        self.depth_texture = self.ctx.texture(
            (int(self.window_size[0] * self.tracing_samples), int(self.window_size[1] * self.tracing_samples)),
            components=1,
            dtype="f4",
        )

        self.screen = quad_fs()

        # camera movement
        self.camera_position = np.array([1, 1, 0], dtype="f4")
        self.camera_direction = np.array([0, 0, 1], dtype="f4")
        self.camera_right = np.cross(self.camera_direction, (0, 1, 0))
        self.yaw = 0
        self.pitch = 0
        self.pressed_keys = set()
        self.mouse_sensitivity = 0.5
        self.speed = 0.01

        # some example data:
        self.spheres = self.ctx.buffer(
            np.array([[0.0, 0.0, 0.0, 0.5 * 0.5], [-0.5, 0.0, 0.5, 0.5 * 0.5], [5.0, 1.0, 15, 2]], dtype="f4").flatten()
        )
        self.planes = self.ctx.buffer(np.array([0.0, 1.0, 0, 0, 0.0, -0.5, 0.0, 0], dtype="f4"))

        self._camera = self.ctx.buffer(
            self.camera_creation(self.camera_position, self.camera_position + self.camera_direction)
        )

        self.show_depth = False

    @staticmethod
    def camera_creation(eye, center=(0, 0, 0), up=(0, 1, 0), fov=45, aspect=16 / 9, aperture=0.01, focus_distance=1.0):
        lens_radius = aperture / 2
        theta = fov * math.pi / 180
        half_height = math.tan(theta / 2.0)
        half_width = aspect * half_height

        origin = eye
        w = normalize(eye - center)
        u = normalize(np.cross(up, w))
        v = np.cross(w, u)

        lower_left_corner = (
            origin - half_width * u * focus_distance - half_height * v * focus_distance - w * focus_distance
        )
        horizontal = 2 * half_width * u * focus_distance
        vertical = 2 * half_height * v * focus_distance

        padding = np.array([0.0])

        return np.concatenate(
            (
                eye,
                padding,
                lower_left_corner,
                padding,
                horizontal,
                padding,
                vertical,
                padding,
                origin,
                padding,
                u,
                padding,
                v,
                padding,
                w,
                padding,
                np.array([lens_radius]),
            ),
            axis=0,
        ).astype("f4")

    def render(self, time: float, frame_time: float) -> None:
        # move the camera
        self.update_camera()

        # bind camera
        self._camera.bind_to_uniform_block(binding=2)
        # bind primitive buffers
        self.spheres.bind_to_storage_buffer(binding=3)
        self.planes.bind_to_storage_buffer(binding=4)
        # bind texture
        self.screen_texture.bind_to_image(0, read=False, write=True)
        self.depth_texture.bind_to_image(1, read=False, write=True)
        self.program.run(self.num_work_groups[0], self.num_work_groups[1], 1)

        # show depth texture
        if self.show_depth:
            self.depth_texture.use(0)
            self.screen.render(self.depth_view_program)
        else:
            # show image texture
            self.screen_texture.build_mipmaps(0, self.mipmap_levels)
            self.screen_texture.use(0)
            self.screen.render(self.view_program)

    def update_camera(self):
        keys = self.wnd.keys
        if self.pressed_keys & {keys.A, keys.D, keys.W, keys.S, keys.UP, keys.DOWN}:
            if keys.A in self.pressed_keys:  # move left
                self.camera_position += -self.camera_right * self.speed
            if keys.D in self.pressed_keys:  # move right
                self.camera_position += self.camera_right * self.speed

            if keys.W in self.pressed_keys:  # move forward
                self.camera_position += self.camera_direction * self.speed
            if keys.S in self.pressed_keys:  # move back
                self.camera_position += -self.camera_direction * self.speed

            if keys.UP in self.pressed_keys:  # move up
                self.camera_position += (0, 1 * self.speed, 0)
            if keys.DOWN in self.pressed_keys:  # move down
                self.camera_position += (0, -1 * self.speed, 0)

            self._camera.write(self.camera_creation(self.camera_position, self.camera_position + self.camera_direction))

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            self.pressed_keys.add(key)

            if key == keys.F:
                self.show_depth = not self.show_depth

        if action == keys.ACTION_RELEASE:
            self.pressed_keys.remove(key)

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):

        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.camera_direction = np.array(
            [
                math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
                math.sin(math.radians(self.pitch)),
                math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            ],
            dtype="f4",
        )
        self.camera_right = np.cross(self.camera_direction, (0., 1., 0.))
        self._camera.write(self.camera_creation(self.camera_position, self.camera_position + self.camera_direction))


if __name__ == "__main__":
    RayTracer.run()
