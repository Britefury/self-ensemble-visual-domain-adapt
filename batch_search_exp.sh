# Mean teacher; augmentation: translation and horizontal flips
python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_meanteacher/log_mnist_usps_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --rampup=80 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=usps_mnist --log_file=results_exp_meanteacher/log_usps_mnist_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --rampup=80 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=mnist_svhn_rgb --log_file=results_exp_meanteacher/log_mnist_svhn_rgb_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --num_epochs=300 --rampup=80 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_meanteacher/log_svhn_mnist_rgb_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --rampup=80 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_meanteacher/log_svhn_mnist_rgb_meanteacher_clsbal_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --num_epochs=300 --rampup=80 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=cifar_stl --log_file=results_exp_meanteacher/log_cifar_stl_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --rampup=80 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=stl_cifar --log_file=results_exp_meanteacher/log_stl_cifar_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --rampup=80 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=syndigits_svhn --log_file=results_exp_meanteacher/log_syndigits_svhn_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --rampup=80 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=synsigns_gtsrb --log_file=results_exp_meanteacher/log_synsigns_gtsrb_meanteacher_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --rampup=80 --device=${1}


# Confidence thresholding; augmentation: translation and horizontal flips
python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_confthresh/log_mnist_usps_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=usps_mnist --log_file=results_exp_confthresh/log_usps_mnist_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=mnist_svhn_rgb --log_file=results_exp_confthresh/log_mnist_svhn_rgb_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_confthresh/log_svhn_mnist_rgb_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_confthresh/log_svhn_mnist_rgb_confthresh_clsbal_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=cifar_stl --log_file=results_exp_confthresh/log_cifar_stl_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=stl_cifar --log_file=results_exp_confthresh/log_stl_cifar_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=syndigits_svhn --log_file=results_exp_confthresh/log_syndigits_svhn_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=synsigns_gtsrb --log_file=results_exp_confthresh/log_synsigns_gtsrb_confthresh_run${2}.txt --src_affine_std=0.0 --tgt_affine_std=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}


# Confidence thresholding; augmentation: minimal (gaussian noise only)
python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_confthresh_minaug/log_mnist_usps_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=usps_mnist --log_file=results_exp_confthresh_minaug/log_usps_mnist_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=mnist_svhn_rgb --log_file=results_exp_confthresh_minaug/log_mnist_svhn_rgb_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_confthresh_minaug/log_svhn_mnist_rgb_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_confthresh_minaug/log_svhn_mnist_rgb_confthresh_clsbal_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=cifar_stl --log_file=results_exp_confthresh_minaug/log_cifar_stl_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=stl_cifar --log_file=results_exp_confthresh_minaug/log_stl_cifar_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=syndigits_svhn --log_file=results_exp_confthresh_minaug/log_syndigits_svhn_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=synsigns_gtsrb --log_file=results_exp_confthresh_minaug/log_synsigns_gtsrb_confthresh_run${2}.txt --src_affine_std=0.0 --src_xlat_range=0.0 --tgt_affine_std=0.0 --tgt_xlat_range=0.0 --standardise_samples --batch_size=64 --cls_balance=0.0 --num_epochs=300 --device=${1}


# Confidence thresholding; augmentation: translation, horizontal flips and affine
python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp_ctaa/log_mnist_usps_ctaa_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=usps_mnist --log_file=results_exp_ctaa/log_usps_mnist_ctaa_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=mnist_svhn_rgb --log_file=results_exp_ctaa/log_mnist_svhn_rgb_ctaa_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_ctaa/log_svhn_mnist_rgb_ctaa_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_ctaa/log_svhn_mnist_rgb_ctaa_clsbal_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=cifar_stl --log_file=results_exp_ctaa/log_cifar_stl_ctaa_run${2}.txt --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=stl_cifar --log_file=results_exp_ctaa/log_stl_cifar_ctaa_run${2}.txt --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=syndigits_svhn --log_file=results_exp_ctaa/log_syndigits_svhn_ctaa_run${2}.txt --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=synsigns_gtsrb --log_file=results_exp_ctaa/log_synsigns_gtsrb_ctaa_run${2}.txt --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}


# Confidence thresholding; augmentation: translation, horizontal flips, affine and mnist <-> svhn specific intensity augmentation
python experiment_domainadapt_meanteacher.py --exp=mnist_svhn_rgb --log_file=results_exp_ctsa/log_mnist_svhn_rgb_ctsa_run${2}.txt --standardise_samples --src_intens_flip --src_intens_scale_range=0.25:1.5 --src_intens_offset_range=-0.5:0.5 --tgt_intens_flip --tgt_intens_scale_range=0.25:1.5 --tgt_intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_ctsa/log_svhn_mnist_rgb_ctsa_run${2}.txt --standardise_samples --src_intens_flip --src_intens_scale_range=0.25:1.5 --src_intens_offset_range=-0.5:0.5 --tgt_intens_flip --tgt_intens_scale_range=0.25:1.5 --tgt_intens_offset_range=-0.5:0.5 --batch_size=256 --cls_balance=0.0 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist_rgb --log_file=results_exp_ctsa/log_svhn_mnist_rgb_ctsa_clsbal_run${2}.txt --standardise_samples --src_intens_flip --src_intens_scale_range=0.25:1.5 --src_intens_offset_range=-0.5:0.5 --tgt_intens_flip --tgt_intens_scale_range=0.25:1.5 --tgt_intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}


