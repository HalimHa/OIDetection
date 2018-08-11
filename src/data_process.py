





def get_data_from_files(root_path):
    

    my_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(root_path,
                                                                                            shuffle=False,
                                                                                               batch_size=batch_size)
    n_rounds = math.ceil(my_generator.samples / my_generator.batch_size)  # size of an epoch
    filenames = my_generator.filenames
     
    my_generator = GeneratorEnqueuer(my_generator)
    my_generator.start()
    my_generator = my_generator.get()
        
    return my_generator