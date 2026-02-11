#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Args (参数处理)
# * ---------------------------------------------------------------------------------------------------------------- * #

function GetArgs() {
  local exp=
  local exp_name=
  local extra_name=
  # shared (通用参数)
  local data_src=celebtalk
  local speaker=
  local wanna_fps=30
  local avoffset_ms=0  # control output avoffset, if not 0, should match speaker's data (控制输出音视频偏移，如果不为0，应匹配说话者的数据)
  local correct_avoffset=true
  # for generating (生成相关参数)
  local is_gen=false
  local test_media=
  local vis_data_src=
  local vis_speaker=
  local same_idle=false
  local dump_offsets=false
  local dump_metrics=false
  local dump_audio=false
  local video_postfix=''
  # other (其他参数)
  local other=
  for i in "$@"; do
    case $i in
      --exp=*          ) exp=${i#*=}            ;; # 实验类型
      --exp_name=*     ) exp_name=${i#*=}       ;; # 实验名称
      --extra_name=*   ) extra_name=${i#*=}     ;; # 额外名称后缀
      # shared
      --data_src=*     ) data_src=${i#*=}       ;; # 数据源
      --speaker=*      ) speaker=${i#*=}        ;; # 说话人
      --wanna_fps=*    ) wanna_fps=${i#*=}      ;; # 期望帧率
      --avoffset_ms=*  ) avoffset_ms=${i#*=}    ;; # 音视频偏移毫秒数
      --keep_avoffset  ) correct_avoffset=false ;; # 保持原有偏移，不修正
      # generating
      --generating     ) is_gen=true            ;; # 生成模式
      --test_media=*   ) test_media=${i#*=}     ;; # 测试媒体文件
      --vis_data_src=* ) vis_data_src=${i#*=}   ;; # 可视化数据源
      --vis_speaker=*  ) vis_speaker=${i#*=}    ;; # 可视化说话人
      --same_idle      ) same_idle=true         ;; # 使用相同的闲置状态
      --dump_offsets   ) dump_offsets=true      ;; # 导出偏移量
      --dump_metrics   ) dump_metrics=true      ;; # 导出指标
      --dump_audio     ) dump_audio=true        ;; # 导出音频
      --video_postfix=*) video_postfix=${i#*=}  ;; # 视频文件名后缀
      # other
      *) other="${other} ${i}";;
    esac
  done

  # > required (必填参数检查)
  [ -n "${exp}"      ] || { echo "[ERROR] --exp is not given!"; exit 1; }
  [ -n "${exp_name}" ] || { echo "[ERROR] --exp_name is not given!"; exit 1; }
  if [[ ! "${exp_name}" =~ .*"${exp}".* ]]; then
    echo "[ERROR] --exp_name '${exp_name}' not match with --exp '${exp}'"; exit 1;
  fi

  # > set default values depends on other (设置默认值)
  [ -n "${test_media}"   ] || { test_media="training_${data_src}"; }
  [ -n "${vis_data_src}" ] || { vis_data_src=${data_src};          }
  [ -n "${vis_speaker}"  ] || { vis_speaker=${speaker};            }
  if [ "$same_idle" == "true" ]; then
    video_postfix+='-same_idle'
  fi

  # > some values depends on mode (根据模式设置参数)
  local max_duration=20
  if [ "$is_gen" == "true" ]; then
    max_duration=1000000
  fi

  # > speakers from vocaset (VOCASET 说话人设置)
  local speakers_voca='${data.VOCASET.REAL3D_SPEAKERS}'
  if [[ "$exp" == "track" ]]; then
    speakers_voca='[]'
  fi

  # > get exp_dir (实验目录)
  local root='${path.runs_dir}/anime'

  # > args string (构建参数字符串)
  local shared=" experiment=animnet_${exp} path.exp_dir=${root}/$speaker/${exp_name}${extra_name}"
  # data
  shared+=" data=all_vocaset dataset=talk_voca datamodule=talk_voca"
  shared+=" data.audio.sliding_window.deepspeech=16"
  shared+=" dataset.data_src_talk=${data_src}"
  shared+=" dataset.speakers_talk=[${speaker}]"
  shared+=" dataset.speakers_voca=${speakers_voca}"
  shared+=" dataset.random_shift_alpha=true"
  shared+=' +dataset.VOCASET=${data.VOCASET}'
  shared+=" datamodule.batch_size=4"
  shared+=" datamodule.train_source_voca=[vocaset_all.csv]"
  # training
  shared+=" trainer.max_epochs=60"
  shared+=" callbacks.model_checkpoint.monitor=epoch-val_metric/mvd-avg:lips"
  # model
  shared+=" model.wanna_fps=${wanna_fps}"
  shared+=" model.data_info.correct_avoffset=${correct_avoffset}"
  shared+=" model.neural_renderer.load_from.path=dont_build"
  shared+=" +model.dont_render=true"
  # loss
  shared+=" model.loss.mesh.part=face_noeyeballs"
  # testing
  shared+=" test_media@model.test_media=${test_media}"
  shared+=" model.test_media.do=[misc]"
  # visualizer
  shared+=" ++model.visualizer.data_src=${vis_data_src}"
  shared+=" ++model.visualizer.speaker=${vis_speaker}"
  shared+=" ++model.visualizer.avoffset_ms=${avoffset_ms}"
  shared+=" ++model.visualizer.video_1st_epoch=true"
  shared+=" ++model.visualizer.video_gap_epochs=10"
  shared+=" ++model.visualizer.draw_gap_steps=1000"
  shared+=" ++model.visualizer.same_idle=${same_idle}"
  shared+=" ++model.visualizer.video_postfix=${video_postfix}"
  shared+=" ++model.visualizer.max_duration=${max_duration}"
  shared+=" ++model.visualizer.generate_test=true"
  shared+=" ++model.visualizer.generate_valid=false"
  shared+=" ++model.visualizer.generate_train=false"
  shared+=" ++model.visualizer.dump_offsets=${dump_offsets}"
  shared+=" ++model.visualizer.dump_metrics=${dump_metrics}"
  shared+=" ++model.visualizer.dump_audio=${dump_audio}"
  shared+=' ++model.visualizer.dkwargs.style_id="${index_of:${dataset.speakers},${model.visualizer.speaker}}"'

  # > Extra args for decmp (decmp 实验的额外参数)
  if [ "$exp" == "decmp" ]; then
    shared="${shared} $(GetArgsForDecmp)"
  fi

  # > Concate with other unknown args (连接其他未知参数)
  echo "${shared} ${other}"
}

function GetArgsForDecmp() {
  local shared=''
  shared+=" dataset.gt_for_swap=true"
  shared+=' +dataset.default_swap=${model.test_media.default_extra}'
  shared+=" datamodule.sampler=BatchSamplerWithDtwPair"
  shared+=" datamodule.sampler_kwargs.pairs_from_same_speaker=true"
  shared+=" datamodule.sampler_kwargs.pairs_from_diff_speaker=true"
  shared+=' +model.datamodule_sampler_kwargs=${datamodule.sampler_kwargs}'
  shared+=" ++model.visualizer.video_gap_epochs=10"
  shared+=" ++model.visualizer.draw_gap_steps=2000"
  echo ${shared}
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Train AnimNet (训练 AnimNet 模型)
# * ---------------------------------------------------------------------------------------------------------------- * #

# * Baselines (Ablation of structure) (基线模型 - 结构消融)
function TrainAnimNet() {
  # * Baseline1: Only Tracked data (基线1：仅使用跟踪数据)
  python3 -m src mode=train $(GetArgs --exp=track --exp_name=animnet-track "$@");
  # * Baseline2: VOCASET + Tracked data (基线2：VOCASET + 跟踪数据)
  python3 -m src mode=train $(GetArgs --exp=cmb3d --exp_name=animnet-cmb3d "$@");
}

function TrainAnimNetTrack() {
  python3 -m src mode=train $(GetArgs --exp=track --exp_name=animnet-track "$@");
}

# * Ablation of structure: naive style encoder from renference animation (结构消融：来自参考动画的简单风格编码器)
function TrainAnimNetRefer() {
  python3 -m src mode=train $(GetArgs --exp=refer --exp_name=animnet-refer "$@") \
    model.data_info.pad_tgt=true \
    dataset.gt_for_swap=true \
    dataset.swap_using_self=true \
    +dataset.default_swap='${model.test_media.default_extra}' \
  ;
}

function TrainAnimNetCmb3dDecmp() {
  python3 -m src mode=train $(GetArgs --exp=cmb3d --exp_name=animnet-cmb3d "$@");
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg";
}

function TrainAnimNetReferE2e() {
  TrainAnimNetRefer "$@";
  TrainAnimNetDecmpAblation "$@" --ablations="e2e";
}

function TrainAnimNetDecmpAblDtwSwp() {
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg no_dtw";
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg no_swp";
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Train Decmp (训练解耦模型)
# * ---------------------------------------------------------------------------------------------------------------- * #

function TrainAnimNetDecmp() {
  python3 -m src mode=train $(GetArgs --exp=decmp --exp_name=animnet-decmp "$@");
}

# * Loss ablations (损失函数消融实验)
function TrainAnimNetDecmpAblation() {
  local ablations=
  local other=
  for i in "$@"; do
    case $i in
      --ablations=*  ) ablations=${i#*=} ;;
      *) other="${other} ${i}"           ;;
    esac
  done

  local extra_args=
  local abl_name=
  for abl in $ablations; do
    [ -n "${abl_name}" ] && { abl_name="${abl_name}_"; }
    abl_name="${abl_name}${abl}"
    case $abl in
      no_dtw) extra_args+=" model.ablation.${abl}=true datamodule.sampler_kwargs.pairs_from_diff_speaker=false" ;;
      no_swp) extra_args+=" model.ablation.${abl}=true model.loss.swp=0" ;;
      no_ph2) extra_args+=" model.ablation.${abl}=true model.loss.cyc=0" ;;
      no_cyc) extra_args+=" model.ablation.${abl}=true model.loss.cyc=0" ;;
      no_reg) extra_args+=" model.ablation.${abl}=true model.loss.reg_z_ctt_close=0 model.loss.reg_z_sty_close=0 model.loss.reg_z_aud_close=0" ;;
      e2e   ) extra_args+=" model.ablation.${abl}=true dataset.gt_for_swap=true +dataset.swap_for_e2e=true datamodule.sampler_kwargs.pairs_from_diff_speaker=false datamodule.sampler_kwargs.pairs_from_same_speaker=false model.loss.swp=0 model.loss.cyc=0 model.loss.reg_z_ctt_close=0 model.loss.reg_z_sty_close=0 model.loss.reg_z_aud_close=0" ;;
    esac
  done

  local args=$(GetArgs --exp=decmp --exp_name=animnet-decmp-abl_${abl_name} ${other})
  python3 -m src mode=train ${args} ${extra_args};
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Generating (生成)
# * ---------------------------------------------------------------------------------------------------------------- * #

function GenAnimNet() {
  local load=
  local other="--generating"
  for i in "$@"; do
    case $i in
      --load=*) load=${i#*=} ;; # 加载的检查点路径
      *) other="${other} ${i}";;
    esac
  done

  [ -n "$load" ] || { echo "--load is not set!"; exit 1; }

  local args=$(GetArgs $other);
  python3 -m src mode=generate hydra.run.dir=.snaps/run \
    ${args} \
    load_from='${path.exp_dir}/checkpoints/'${load} \
    strict=false \
  ;
}

function GenAll() {
  local load_decmp=
  local load_cmb3d=
  local load_refer=
  local load_track=
  local ablation=
  local shared=
  for i in "$@"; do
    case $i in
      --load_decmp=*) load_decmp=${i#*=} ;; # 加载解耦模型
      --load_cmb3d=*) load_cmb3d=${i#*=} ;; # 加载组合3D模型
      --load_refer=*) load_refer=${i#*=} ;; # 加载参考模型
      --load_track=*) load_track=${i#*=} ;; # 加载跟踪模型
      --ablation    ) ablation=true       ;; # 是否进行消融实验
      *) shared="${shared} ${i}";;
    esac
  done

  [ -n "${load_cmb3d}" ] && GenAnimNet --exp='cmb3d' --exp_name='animnet-cmb3d' --load=${load_cmb3d} ${shared};
  [ -n "${load_refer}" ] && GenAnimNet --exp='refer' --exp_name='animnet-refer' --load=${load_refer} ${shared};
  [ -n "${load_track}" ] && GenAnimNet --exp='track' --exp_name='animnet-track' --load=${load_track} ${shared};

  if [ -n "${load_decmp}" ]; then
    GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load=${load_decmp} ${shared};
    if [ -n "${ablation}" ]; then
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_dtw' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_swp' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_cyc' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_e2e'           --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_ph2' --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_ph2' --load=${load_decmp} ${shared};
    fi
  fi
}
