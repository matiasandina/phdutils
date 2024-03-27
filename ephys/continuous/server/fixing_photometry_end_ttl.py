file = ["/synology-nas/MLA/beelink1/MLA148/2023-12-03/ttl/sub-MLA148_ses-20231203T144807_ttl_in.bin"]
ttl2, ttl2_samples = read_stack_chunks(file, 8, np.int8, True)
ttl2_array = normalize_ttl(ttl2, method="max")

pulse_onset = np.where(np.diff(ttl2_array[0,:], prepend=0) > 0)[0]

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ttl2_array[0,:])
plt.show(block = False)


block1_dir = ["/synology-nas/MLA/TDT/PhotoOptoRandTrialsTTL2Box-230412-115924/MLA147148-231203-110016/"]
block2_dir = ["/synology-nas/MLA/TDT/PhotoOptoRandTrialsTTL2Box-230412-115924/MLA147148-231203-150659/"]

import tdt

block1 = tdt.read_block(block1_dir[0], t1=0, t2=1)
block2 = tdt.read_block(block2_dir[0], t1=0, t2=1)

block1_onset = tdt.read_block(block1_dir[0], store = fix_tdt_names(pulse_sync_name, '/')).epocs[fix_tdt_names(pulse_sync_name, '_')].onset
block2_onset = tdt.read_block(block2_dir[0], store = fix_tdt_names(pulse_sync_name, '/')).epocs[fix_tdt_names(pulse_sync_name, '_')].onset

block1_timestamps = block1.info.start_date

import polars as pl
block1_df = pl.DataFrame({"onset_sec": block1_onset})

block1_df = block1_df.with_columns(
    pl.col("onset_sec").apply(
        lambda x: 
        datetime.timedelta(seconds = x) + 
        block1.info.start_date).alias("datetime")
        )

block2_df = pl.DataFrame({"onset_sec": block2_onset})
block2_df = block1_df.with_columns(
    pl.col("onset_sec").apply(
        lambda x: 
        datetime.timedelta(seconds = x) + 
        block2.info.start_date).alias("datetime")
        )

df = pl.concat([block1_df, block2_df])

plt.figure()
plt.plot(df['datetime'].to_numpy(), np.repeat(1,df.shape[0]), 'k|')
plt.show()