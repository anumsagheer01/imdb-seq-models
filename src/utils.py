import os
def hardware_report():
    ram_kb = int([l for l in open('/proc/meminfo') if l.startswith('MemTotal:')][0].split()[1])
    ram_gb = round(ram_kb/1e6, 2)
    try:
        import tensorflow as tf
        accel = "GPU" if tf.config.list_physical_devices('GPU') else "CPU only"
    except:
        accel = "CPU only"
    return dict(cpu_cores=os.cpu_count(), ram_gb=ram_gb, accelerator=accel)
