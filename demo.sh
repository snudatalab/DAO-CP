if [ $# -ne 1 ]
then
    echo "Choose a dataset for demo (synthetic, video, stock, hall, korea)"
    echo "Usage: $0 <dataset_name>"
    exit
fi

cd src; python main.py $1
