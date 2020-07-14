#!/bin/bash
#
# To start members:
# 	digits_members deploy <number_of_members>
# 
# To terminate members
#   digits_members delete 
#



cp $(pwd)/config/member.yaml $(pwd)/digits-example-project/project/project.yaml

command=$1
members=$2
if [ "$1" == "deploy" ]; then  
	name="digits_member_"
	for i in $(seq 1 $members);
		do
			docker run -d --name ${name}_$i -e ACCESS_KEY="${name}$i" --network host -v $(pwd)/mnist_data:/app/data -v $(pwd)/digits-example-project/project:/app/project scaleoutsystems/member
			echo Deployed member ${name}_$i with ID ${name}$i
		done
elif [ "$1" == "delete" ]
then 
	docker ps -a | grep "digits_member_*" | awk '{print $1}' | xargs docker stop
	docker ps -a | grep "digits_member_*" | awk '{print $1}' | xargs docker rm 
else
	echo "Unknown command" $1
fi