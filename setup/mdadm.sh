mdadm --stop /dev/md123
mdadm --stop /dev/md124
mdadm --stop /dev/md125
mdadm --stop /dev/md126
mdadm --stop /dev/md127
mdadm --zero-superblock /dev/nvme{1,2}n1p2
mdadm --zero-superblock /dev/nvme{1,2,3,4}n1p3
mdadm --zero-superblock /dev/nvme{1,2,3,4,5,6,7,8,9,10}n1p4
mdadm --zero-superblock /dev/nvme{3,4,5,6,7,8,9,10}n1p2
mdadm --zero-superblock /dev/nvme{5,6,7,8,9,10}n1p3

mdadm --create /dev/md2 --level=0 --raid-devices=2 /dev/nvme{1,2}n1p2
mdadm --create /dev/md4 --level=0 --raid-devices=4 /dev/nvme{1,2,3,4}n1p3
mdadm --create /dev/md8 --level=0 --raid-devices=8 /dev/nvme{3,4,5,6,7,8,9,10}n1p2
mdadm --create /dev/md6 --level=0 --raid-devices=6 /dev/nvme{5,6,7,8,9,10}n1p3
mdadm --create /dev/md10 --level=0 --raid-devices=10 /dev/nvme{1,2,3,4,5,6,7,8,9,10}n1p4

mkfs.ext4 /dev/md2
mkfs.ext4 /dev/md4
mkfs.ext4 /dev/md6
mkfs.ext4 /dev/md8
mkfs.ext4 /dev/md10


