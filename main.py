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


class RayTracer(mglw.WindowConfig):
    # moderngl_window settings
    gl_version = (4, 3)
    title = "RayTracing demo"
    resource_dir = (Path(__file__) / "../resources").absolute()
    window_size = 720, 720
    aspect_ratio = None
    tracing_samples = 2
    near_far_planes = (0.0, 65536.0)
    mipmap_levels = math.floor(math.log2(tracing_samples))
    resizable = False
    clear_color = 51 / 255, 51 / 255, 51 / 255

    group_size = (32, 32)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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

        # self.screen_texture = self.load_texture_2d('test-pattern.png')
        self.screen = quad_fs()

        # some example data:
        self.spheres = self.ctx.buffer(np.array([0.0, 0.0, 0.0, 0.5 * 0.5, -0.5, 0.0, 0.25, 0.5 * 0.5], dtype="f4"))
        self.planes = self.ctx.buffer(np.array([0.0, 1.0, 0, 0, 0.0, -0.5, 0.0, 0], dtype="f4"))

        cam_pos = np.array([0, 0, -5.0])

        cam_proj = Matrix44.perspective_projection(90, 1, 1, 2, dtype="f4").inverse

        ray00 = pyrr.matrix44.apply_to_vector(vec=np.array([-0.5, -0.5, 0, 1], dtype="f4"), mat=cam_proj)
        ray00 /= ray00[-1]
        ray00 = ray00[:-1]
        ray00 -= cam_pos

        ray10 = pyrr.matrix44.apply_to_vector(vec=np.array([+0.5, -0.5, 0, 1], dtype="f4"), mat=cam_proj)
        ray10 /= ray00[-1]
        ray10 = ray10[:-1]
        ray10 += cam_pos

        ray11 = pyrr.matrix44.apply_to_vector(vec=np.array([+0.5, +0.5, 0, 1], dtype="f4"), mat=cam_proj)
        ray11 /= ray00[-1]
        ray11 = ray11[:-1]
        ray11 += cam_pos

        ray01 = pyrr.matrix44.apply_to_vector(vec=np.array([-0.5, +0.5, 0, 1], dtype="f4"), mat=cam_proj)
        ray01 /= ray00[-1]
        ray01 = ray01[:-1]
        ray01 += cam_pos

        buff_data = np.array(
            [*cam_pos.tolist(), 0, *ray00.tolist(), 0, *ray01.tolist(), 0, *ray10.tolist(), 0, *ray11.tolist(), 0],
            dtype="f4",
        ).flatten()
        self.camera = self.ctx.buffer(buff_data)

        self.show_depth = False

    def render(self, time: float, frame_time: float) -> None:
        # render the vao
        # bind camera
        self.camera.bind_to_uniform_block(binding=2)
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

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.F:
                self.show_depth = not self.show_depth


if __name__ == "__main__":
    RayTracer.run()
