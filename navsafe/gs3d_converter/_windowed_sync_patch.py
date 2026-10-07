"""Perf patch: window py123d NuplanLogParser.iter_modalities_sync to [t0,t1].

py123d walks EVERY lidar_pc of the whole nuPlan log and extracts boxes + cameras
+ lidar per frame, then _WindowedNuplanLogParser discards all but the scene
window [t0,t1]. For long logs (e.g. 26 min / 3.5M boxes) that is minutes of
wasted ORM work AND per-frame CephFS sensor reads. This replaces the method with
a faithful copy that skips frames outside the active window (set per-scene via
_SYNC_WINDOW by _WindowedNuplanLogParser). In-window frames are processed
identically, so the output arrow is unchanged; out-of-window frames were
discarded by the wrapper anyway. When no window is active, the original runs.
"""
import threading

_SYNC_WINDOW = threading.local()


def _install():
    import py123d.parser.nuplan.nuplan_parser as m

    if getattr(m.NuplanLogParser.iter_modalities_sync, "_windowed", False):
        return
    _orig = m.NuplanLogParser.iter_modalities_sync

    def iter_modalities_sync(self):
        win = getattr(_SYNC_WINDOW, "win", None)
        if win is None:
            yield from _orig(self)
            return
        _t0, _t1 = win
        ego_md = m.NUPLAN_EGO_STATE_SE3_METADATA
        box_md = m.NUPLAN_BOX_DETECTIONS_SE3_METADATA
        cam_mds = m._get_nuplan_camera_metadata(self._source_log_path, self._nuplan_sensor_root)
        lidar_md = m._get_nuplan_lidar_merged_metadata(self._nuplan_sensor_root, self._source_log_path.stem)
        db = m.NuPlanDB(str(self._nuplan_data_root), str(self._source_log_path), None)
        try:
            step_interval = int(m.TARGET_DT / m.NUPLAN_DEFAULT_DT)
            offset = m._get_ideal_lidar_pc_offset(self._source_log_path, db)
            num_steps = len(db.lidar_pc)
            for i in range(offset, num_steps, step_interval):
                pc = db.lidar_pc[i]
                ts = int(pc.timestamp)
                if ts < _t0:
                    del pc
                    continue
                if ts > _t1:
                    break
                token = pc.token
                timestamp = m.Timestamp.from_us(pc.timestamp)
                ego = m._extract_nuplan_ego_state(pc, ego_md)
                boxes = m._extract_nuplan_box_detections(pc, self._source_log_path, timestamp, box_md)
                tls = m._extract_nuplan_traffic_lights(db, token, timestamp)
                cams = m._extract_nuplan_cameras(lidar_pc_token=token, source_log_path=self._source_log_path, nuplan_sensor_root=self._nuplan_sensor_root, metadatas=cam_mds)
                lidar = m._extract_nuplan_lidar_data(nuplan_lidar_pc=pc, nuplan_sensor_root=self._nuplan_sensor_root, metadata=lidar_md)
                custom = m._extract_nuplan_scenario_data(pc)
                mods = [ego, boxes, tls, custom]
                mods.extend(cams)
                if lidar is not None:
                    mods.append(lidar)
                yield m.ModalitiesSync(timestamp=timestamp, modalities=mods)
                del pc
        finally:
            db.detach_tables()
            db.remove_ref()
            del db

    iter_modalities_sync._windowed = True
    m.NuplanLogParser.iter_modalities_sync = iter_modalities_sync


_install()
