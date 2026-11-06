""" Shared pytest settings: the reference reconstruction is marked slow. """


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: runs the reference reconstruction, '
                                       'about ten seconds; skip with -m "not slow"')
