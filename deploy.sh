#!/bin/bash
# 1. clone repository
 git clone https://github.com/TheHoboCoder/deploy-geodjango.git ../deploy-geodjango
# 2. remove all in modules just in case
mkdir modules || rm -R modules/*
cd ../deploy-geodjango/
chmod u+x deploy.sh common_funcs.sh
pwd
# 3. run script
./deploy.sh geoportal geoportal.com
# 4. change ownership of media
chown geoportal:www-data /webapps/geoportal_project/geoportal/modules
chmod g+w /webapps/geoportal_project/geoportal/modules
echo "DEPLOYMENT DONE!!!"