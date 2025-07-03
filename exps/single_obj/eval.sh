CONTAINER=$SRCROOT/env.sif
CONFIGDIR=exps/single_obj
BASEEXPDIR=exps/single_obj/logs
mkdir -p $BASEEXPDIR

RUN="apptainer run --nv $CONTAINER python"


#EXPDIR=$BASEEXPDIR/stage0_cv
#mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/single_obj_stage0.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_cv.py $EXPDIR

#EXPDIR=$BASEEXPDIR/stage0_crtv
#mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/single_obj_stage0.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_ctrv.py $EXPDIR


#EXPDIR=$BASEEXPDIR/stage1_cv
#mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/single_obj_stage1.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_cv.py $EXPDIR

#EXPDIR=$BASEEXPDIR/stage1_crtv
#mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/single_obj_stage1.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_ctrv.py $EXPDIR

EXPDIR=$BASEEXPDIR/stage2_ctrv
mkdir -p $EXPDIR
cp $SRCROOT/checkpoints/single_obj_stage2_ctrv.pt $EXPDIR/checkpoint.pt
$RUN run.py $CONFIGDIR/eval_ctrv.py $EXPDIR

EXPDIR=$BASEEXPDIR/stage2_cv
mkdir -p $EXPDIR
cp $SRCROOT/checkpoints/single_obj_stage2_cv.pt $EXPDIR/checkpoint.pt
$RUN run.py $CONFIGDIR/eval_cv.py $EXPDIR


#EXPDIR=$BASEEXPDIR/sched_fixed
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_fixed.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval_fixed.py $EXPDIR

#EXPDIR=$BASEEXPDIR/sched_greedy
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_greedy.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval_greedy.py $EXPDIR

#EXPDIR=$BASEEXPDIR/sched_train_fixed_eval_greedy
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/sched_fixed/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_greedy.py $EXPDIR

#EXPDIR=$BASEEXPDIR/sched_train_greedy_eval_fixed
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/sched_greedy/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval_fixed.py $EXPDIR



#for mu in 0.05 0.1 0.15 0.2 0.25 0.3; do
    #EXPDIR=$BASEEXPDIR/eval_greedy_mu$mu
    #mkdir -p $EXPDIR
    #cp $BASEEXPDIR/sched_greedy/checkpoint.pt $EXPDIR/checkpoint.pt
    #$RUN run.py $CONFIGDIR/eval_greedy_mu$mu.py $EXPDIR
#done


#EXPDIR=$BASEEXPDIR/calib
#mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/refined_classifier.pt $EXPDIR/checkpoint.pt
#$RUN run_calib.py $CONFIGDIR/train_calib.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval_coco.py $EXPDIR


#EXPDIR=$BASEEXPDIR/cls
##mkdir -p $EXPDIR
#cp $SRCROOT/checkpoints/9_calibrated_scenarios.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_cls.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval.py $EXPDIR
#$RUN eval_coco.py $CONFIGDIR/eval.py $EXPDIR

#EXPDIR=$BASEEXPDIR/sched_train_dets
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_sched.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval_fixed.py $EXPDIR




#EXPDIR=$BASEEXPDIR/tracker
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_tracker.py $EXPDIR
#$RUN run.py $CONFIGDIR/eval.py $EXPDIR

#EXPDIR=$BASEEXPDIR/sweep_pre_tracker
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/sweep.py $EXPDIR

#EXPDIR=$BASEEXPDIR/reboot
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/eval.py $EXPDIR



#EXPDIR=$BASEEXPDIR/clutter_train
#mkdir -p $EXPDIR
#cp $BASEEXPDIR/std/checkpoint.pt $EXPDIR/checkpoint.pt
#$RUN run.py $CONFIGDIR/train_clutter.py $EXPDIR --train_threshold 0.01
#$RUN run.py $CONFIGDIR/sweep.py $EXPDIR


#cp $EXPDIR/checkpoint.pt $EXPDIR/checkpoint_cls.pt
#$RUN run.py $CONFIGDIR/train_std.py $EXPDIR
#cp $EXPDIR/checkpoint.pt $EXPDIR/checkpoint_std.pt

#cp $EXPDIR/checkpoint_cls.pt $EXPDIR/checkpoint.pt
#$RUN eval_coco.py $CONFIGDIR/eval.py $EXPDIR

#$RUN run.py $CONFIGDIR/train_tracker.py $EXPDIR
#cp $EXPDIR/checkpoint.pt $EXPDIR/checkpoint_tracker.pt


#$RUN run.py $CONFIGDIR/eval.py $EXPDIR
