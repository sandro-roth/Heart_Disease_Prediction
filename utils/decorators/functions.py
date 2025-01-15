def memorizer(func):
    '''Decorator to memorize funciton values in recursive use
    func: function to be called'''
    memory = {}
    def wrapper(*args, **kwargs):
        #Define a hashable kwargs-key
        kwargs_key = tuple(sorted(kwargs.items()))
        if (args, kwargs_key) not in memory:
            memory[(args, kwargs_key)] = func(*args, **kwargs)
        return memory[(args, kwargs_key)]
    return wrapper