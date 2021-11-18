%% Simulate data stream from recorded file

cfg                 = [];
cfg.minblocksize    = 0.0;
cfg.maxblocksize    = 1.0;
cfg.channel         = 'all';
cfg.speed           = 1.0;

cfg.source.dataset  = 'Rest_raw.fif';
cfg.target.datafile = 'buffer://localhost:1972';

ft_realtime_fileproxy(cfg);