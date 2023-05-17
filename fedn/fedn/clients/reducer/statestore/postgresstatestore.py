import confuse

def get_config(model,value):
    import SQLAlchemy



from reducerstatestore import ReducerStateStore
class PostgressStateStore(ReducerStateStore):

    @classmethod
    def load_from_file(cls, defaults):
        """
        Load configuration from file.
        """
        config = confuse.Configuration('FEDn_Reducer', __name__)
        config.set_file(defaults)

        cls.state = config['control']['state'].get()
        try:
            cls.transition(cls.state)
        except KeyError:
            cls.transition("idle")

        try:
            cls.model = config['control']['model'].get()
        except Exception as e:
            pass
        cls.round = config['control']['round'].get()

        cls.round_config = {'timeout': 180, 'validate': True}
        try:
            cls.round_config = config['control']['round_config'].get()
        except:
            pass
        cls.set_storage_backend(config['storage'].get())
        #return config
        return True

    def __init__(self, network_id, config, defaults=None):
        self.__inited = False

        if defaults:
            if ReducerStateStore.load_from_file(self, defaults):
                if self.write_config_to_statestore():
                    self.__inited = True

        if not self.__inited:
            self.reducer = get_config('network','reducer')
            self.combiners = get_config('network','combiners')
            self.clients = get_config('network','clients')
            self.storage = get_config('network','storage')
            self.certificates = get_config('network','certificates')

            self.control_config = get_config('control','config')
            self.state = get_config('control','state')
            self.model = get_config('control','model')
            self.round = get_config('control','round')

            self.status = get_config('control','status')
            self.round_time = get_config('control','round_time')
            self.psutil_monitoring = get_config('control','psutil_monitoring')
            self.combiner_round_time = get_config('control','combiner_round_time')

        self.__inited = True

        """Mongo Configuration 
        # FEDn network
        self.network = self.mdb['network']
            self.reducer = self.network['reducer']
            self.combiners = self.network['combiners']
            self.clients = self.network['clients']
            self.storage = self.network['storage']
            self.certificates = self.network['certificates']
        # Control
        self.control = self.mdb['control']
            self.control_config = self.control['config']
            self.state = self.control['state']
            self.model = self.control['model']
            self.round = self.control["round"]

        # Logging and dashboards
            self.status = self.control["status"]
            self.round_time = self.control["round_time"]
            self.psutil_monitoring = self.control["psutil_monitoring"]
            self.combiner_round_time = self.control['combiner_round_time']

        self.__inited = True
        """