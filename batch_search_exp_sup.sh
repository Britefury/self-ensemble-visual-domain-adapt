# Supervised only minimal augmentation
python experiment_sup.py --exp=mnist_usps --log_file=results_exp_sup_noaug/log_mnist_usps_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=usps_mnist --log_file=results_exp_sup_noaug/log_usps_mnist_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_noaug/log_mnist_svhn_rgb_noaug_run${2}.txt --standardise_samples --xlat_range=0.0 --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_noaug/log_svhn_mnist_rgb_noaug_run${2}.txt --standardise_samples --xlat_range=0.0 --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_noaug/log_mnist_svhn_rgb_noaug_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_noaug/log_svhn_mnist_rgb_noaug_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=cifar_stl --log_file=results_exp_sup_noaug/log_cifar_stl_noaug_run${2}.txt --standardise_samples --hflip --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=stl_cifar --log_file=results_exp_sup_noaug/log_stl_cifar_noaug_run${2}.txt --standardise_samples --hflip --affine_std=0.0 --xlat_range=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=syndigits_svhn --log_file=results_exp_sup_noaug/log_syndigits_svhn_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_syndigits --log_file=results_exp_sup_noaug/log_svhn_syndigits_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=128 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=synsigns_gtsrb --log_file=results_exp_sup_noaug/log_synsigns_gtsrb_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=gtsrb_synsigns --log_file=results_exp_sup_noaug/log_gtsrb_synsigns_noaug_run${2}.txt --standardise_samples --affine_std=0.0 --xlat_range=0.0 --batch_size=128 --num_epochs=300 --device=${1}


# Supervised only: translation and horizontal flips
python experiment_sup.py --exp=mnist_usps --log_file=results_exp_sup_tf/log_mnist_usps_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=usps_mnist --log_file=results_exp_sup_tf/log_usps_mnist_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_tf/log_mnist_svhn_rgb_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_tf/log_svhn_mnist_rgb_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_tf/log_mnist_svhn_rgb_tf_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_tf/log_svhn_mnist_rgb_tf_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=cifar_stl --log_file=results_exp_sup_tf/log_cifar_stl_tf_run${2}.txt --standardise_samples --hflip --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=stl_cifar --log_file=results_exp_sup_tf/log_stl_cifar_tf_run${2}.txt --standardise_samples --hflip --affine_std=0.0 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=syndigits_svhn --log_file=results_exp_sup_tf/log_syndigits_svhn_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_syndigits --log_file=results_exp_sup_tf/log_svhn_syndigits_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=128 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=synsigns_gtsrb --log_file=results_exp_sup_tf/log_synsigns_gtsrb_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=gtsrb_synsigns --log_file=results_exp_sup_tf/log_gtsrb_synsigns_tf_run${2}.txt --standardise_samples --affine_std=0.0 --batch_size=128 --num_epochs=300 --device=${1}


# Supervised only: translation, horizontal flips and affine augmentation
python experiment_sup.py --exp=mnist_usps --log_file=results_exp_sup_tfa/log_mnist_usps_tfa_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=usps_mnist --log_file=results_exp_sup_tfa/log_usps_mnist_tfa_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_tfa/log_mnist_svhn_rgb_tfa_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_tfa/log_svhn_mnist_rgb_tfa_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=mnist_svhn_rgb --log_file=results_exp_sup_tfa/log_mnist_svhn_rgb_tfa_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_mnist_rgb --log_file=results_exp_sup_tfa/log_svhn_mnist_rgb_tfa_specaug_run${2}.txt --standardise_samples --intens_flip --intens_scale_range=0.25:1.5 --intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=cifar_stl --log_file=results_exp_sup_tfa/log_cifar_stl_tfa_run${2}.txt --standardise_samples --hflip --batch_size=256 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=stl_cifar --log_file=results_exp_sup_tfa/log_stl_cifar_tfa_run${2}.txt --standardise_samples --hflip --batch_size=256 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=syndigits_svhn --log_file=results_exp_sup_tfa/log_syndigits_svhn_tfa_run${2}.txt --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=svhn_syndigits --log_file=results_exp_sup_tfa/log_svhn_syndigits_tfa_run${2}.txt --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}

python experiment_sup.py --exp=synsigns_gtsrb --log_file=results_exp_sup_tfa/log_synsigns_gtsrb_tfa_run${2}.txt --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}
python experiment_sup.py --exp=gtsrb_synsigns --log_file=results_exp_sup_tfa/log_gtsrb_synsigns_tfa_run${2}.txt --standardise_samples --batch_size=128 --num_epochs=300 --device=${1}
