## Building Docker image

building the image
```
docker build --tag blink_api .
```

running with faiss
```
--faiss_index hnsw --index_path models/faiss_hnsw_index.pkl
```

https://docs.aws.amazon.com/efs/latest/ug/wt1-test.html

sudo apt-get update
sudo apt-get -y install git binutils
mkdir nfs
cd nfs
git clone https://github.com/aws/efs-utils
cd efs-utils
./build-deb.sh
sudo apt-get -y install ./build/amazon-efs-utils*deb



aws ec2 create-security-group \
--region us-east-1 \
--group-name efs-ec2-sg \
--description "Amazon EFS SG for EC2 instance" \
--vpc-id vpc-7def7600

{
    "GroupId": "sg-02269fa59660aa714"
}

aws ec2 create-security-group \
--region us-east-1 \
--group-name efs-mt-sg \
--description "Amazon EFS SG for mount target" \
--vpc-id vpc-7def7600

{
    "GroupId": "sg-0680d8ba9dc34d785"
}

aws ec2 authorize-security-group-ingress \
--group-id sg-02269fa59660aa714 \
--protocol tcp \
--port 22 \
--cidr 0.0.0.0/0 --region us-east-1

aws ec2 authorize-security-group-ingress \
--group-id sg-0680d8ba9dc34d785 \
--protocol tcp \
--port 2049 \
--source-group sg-02269fa59660aa714 \
--region us-east-1

aws efs create-file-system \
--encrypted \
--creation-token FileSystemForBlink \
--tags Key=Name,Value=FileSystemForBlink \
--region us-east-1

{
    "OwnerId": "877271521523",
    "CreationToken": "FileSystemForBlink",
    "FileSystemId": "fs-40e9aff4",
    "FileSystemArn": "arn:aws:elasticfilesystem:us-east-1:877271521523:file-system/fs-40e9aff4",
    "CreationTime": 1627384677.0,
    "LifeCycleState": "creating",
    "Name": "FileSystemForBlink",
    "NumberOfMountTargets": 0,
    "SizeInBytes": {
        "Value": 0,
        "ValueInIA": 0,
        "ValueInStandard": 0
    },
    "PerformanceMode": "generalPurpose",
    "Encrypted": true,
    "KmsKeyId": "arn:aws:kms:us-east-1:877271521523:key/72272e87-f8da-4665-baf6-e0419f708987",
    "ThroughputMode": "bursting",
    "Tags": [
        {
            "Key": "Name",
            "Value": "FileSystemForBlink"
        }
    ]
}

aws efs create-mount-target \
--file-system-id fs-40e9aff4 \
--subnet-id  subnet-b7f7f0b9 \
--security-group sg-0680d8ba9dc34d785 \
--region us-east-1

{
    "OwnerId": "877271521523",
    "MountTargetId": "fsmt-1786e2a2",
    "FileSystemId": "fs-40e9aff4",
    "SubnetId": "subnet-b7f7f0b9",
    "LifeCycleState": "creating",
    "IpAddress": "172.31.74.38",
    "NetworkInterfaceId": "eni-0d42511601ab5ced1",
    "AvailabilityZoneId": "use1-az5",
    "AvailabilityZoneName": "us-east-1f",
    "VpcId": "vpc-7def7600"
}

sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-40e9aff4.efs.us-east-1.amazonaws.com:/ ~/efs-mount-point
sudo chown ubuntu ~/efs-mount-point


aws ec2 authorize-security-group-ingress \
--group-id sg-0680d8ba9dc34d785 \
--protocol tcp \
--port 2049 \
--source-group sg-0fb9bec03a4a1e70b \
--region us-east-1
