import yaml

def config_laoder(config_file):
    config_info = {}
    with open(config_file, "r") as config:
        try: 
          config_info = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
            
          
          
    return config_info

# conf = config_laoder("configs\config.yaml")
# print(conf)

    