%% Simulate data stream from recorded file

addpath ~/fieldtrip-20220711
ft_defaults
cfg                 = [];
cfg.minblocksize    = 0.0;
cfg.maxblocksize    = 1.0;
cfg.channel         = 'all';
cfg.speed           = 1.0;

cfg.source.dataset  = 'C:\\Users\\Simon\\.meg_nfb\\rest.fif';
cfg.target.datafile = 'buffer://localhost:1972';

ft_realtime_fileproxy(cfg);