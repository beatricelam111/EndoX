import omni.usd
import omni.timeline
from pxr import Usd, Gf
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import asyncio

class CameraPoseCapture:
    def __init__(self, camera_prim_path, output_path):
        self.camera_prim_path = camera_prim_path
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = self.timeline.get_time_codes_per_seconds()
        self.start_time = self.timeline.get_start_time()
        self.end_time = self.timeline.get_end_time()
        self.num_frames = int((self.end_time - self.start_time) * self.tcps)
        self.camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)

    @staticmethod
    def extract_pose_at_time(camera_prim):
        """Return (tx, ty, tz, qx, qy, qz, qw) for the given camera prim."""
        matrix = omni.usd.get_world_transform_matrix(camera_prim)
        translation = matrix.ExtractTranslation()
        tx, ty, tz = translation[0], translation[1], translation[2]
        rot_mat = np.array(matrix.ExtractRotationMatrix())
        rot = R.from_matrix(rot_mat)
        qx, qy, qz, qw = rot.as_quat()  # format: x, y, z, w
        return tx, ty, tz, qx, qy, qz, qw

    async def save_pose_file(self):
        start_tc = int(round(self.start_time * self.tcps))
        with open(self.output_path, "w") as f:
            for timecode in range(start_tc, start_tc + self.num_frames):
                self.timeline.set_current_time(timecode / self.tcps)
                await omni.kit.app.get_app().next_update_async()
                tx, ty, tz, qx, qy, qz, qw = self.extract_pose_at_time(self.camera_prim)
                f.write(f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
        print(f"Saved camera poses for {self.num_frames} frames to {self.output_path}")

# --- Main execution ---
if __name__ == "__main__":
    camera_prim_path = "/World/CapsuleCam/Camera"  # Change to your camera path
    output_path = "C:/output/aov_capture/camera_pose.txt"  # Change to your output path
    capturer = CameraPoseCapture(camera_prim_path, output_path)
    asyncio.ensure_future(capturer.save_pose_file())
