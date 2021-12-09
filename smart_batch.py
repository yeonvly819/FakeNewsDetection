
def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval 
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the 
                              course of the for-loop.
    '''
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller. 
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def make_smart_batches(text, label, group_ids, batch_size, max_len, padding_token_id=0):
    '''
    This function combines all of the required steps to prepare batches.
    '''

    print('Creating Smart Batches from {:,} examples with batch size {:,}...\n'.format(len(text), batch_size))
    
    # =========================
    #      Select Batches
    # =========================    

    # Sort the two lists together by the length of the input sequence.
    samples = sorted(zip(text, label, group_ids), key=lambda x: len(x[0]))
    print('{:>10,} samples after sorting\n'.format(len(samples)))

    import random
    import torch

    # List of batches that we'll construct.
    batch_ordered_text = []
    batch_ordered_label = []
    batch_ordered_group_ids = []

    print('Creating batches of size {:}...'.format(batch_size))

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(samples), num_desired_updates=10)
    
    # Loop over all of the input samples...    
    while len(samples) > 0:
        
        # Report progress.
        if ((len(batch_ordered_text) % update_interval) == 0 \
            and not len(batch_ordered_text) == 0):
            print('  Selected {:,} batches.'.format(len(batch_ordered_text)))

        # `to_take` is our actual batch size. It will be `batch_size` until 
        # we get to the last batch, which may be smaller. 
        to_take = min(batch_size, len(samples))

        # Pick a random index in the list of remaining samples to start
        # our batch at.
        select = random.randint(0, len(samples) - to_take)

        # Select a contiguous batch of samples starting at `select`.
        #print("Selecting batch from {:} to {:}".format(select, select+to_take))
        batch = samples[select:(select + to_take)]

        #print("Batch length:", len(batch))

        # Each sample is a tuple--split them apart to create a separate list of 
        # sequences and a list of labels for this batch.
        batch_ordered_text.append([s[0] for s in batch])
        batch_ordered_label.append([s[1] for s in batch])
        batch_ordered_group_ids.append([s[2] for s in batch])

        # Remove these samples from the list.
        del samples[select:select + to_take]

    print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_text)))

    # =========================
    #        Add Padding
    # =========================    

    print('Padding out sequences within each batch...')

    py_en = []
    py_en_masks = []
    # For each batch...
    for batch_txt in batch_ordered_text :

        # New version of the batch, this time with padded sequences and now with
        # attention masks defined.
        batch_padded_txt = []
        batch_masks_txt = []
        
        # First, find the longest sample in the batch. 
        # Note that the sequences do currently include the special tokens!
        en_max_size = max([len(sen) for sen in batch_txt])

        # For each input in this batch...
        for sen_en in batch_txt :
 
            # How many pad tokens do we need to add?
            en_num_pads = en_max_size - len(sen_en)

            # Add `num_pads` padding tokens to the end of the sequence.
            padded_en = sen_en + [padding_token_id]*en_num_pads

            # Define the attention mask--it's just a `1` for every real token
            # and a `0` for every padding token.
            attn_mask_en = [1] * len(sen_en) + [padding_token_id] * en_num_pads

            # Add the padded results to the batch.
            batch_padded_txt.append(padded_en[:max_len])
            batch_masks_txt.append(attn_mask_en[:max_len])

        # Our batch has been padded, so we need to save this updated batch.
        # We also need the inputs to be PyTorch tensors, so we'll do that here.
        # Todo - Michael's code specified "dtype=torch.long"
        py_en.append(torch.LongTensor(batch_padded_txt))
        py_en_masks.append(torch.LongTensor(batch_masks_txt))
    
    print('  DONE.')

    # Return the smart-batched dataset!
    return (py_en, py_en_masks, batch_ordered_label, batch_ordered_group_ids)
