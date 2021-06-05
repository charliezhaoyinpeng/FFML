from plottingFunctions import *
import copy, random, csv

if __name__ == "__main__":
    dataname = 'Adult - 0.9 Validation'
    start_task = 0
    num_tasks = 41
    every = 1

    GenOLC1 = r'results_GenOLC1'
    GenOLC2 = r'results_GenOLC2'
    GenOLC3 = r'results_GenOLC3'

    TWP1 = r'results_TWP1'
    TWP2 = r'results_TWP2'
    TWP3 = r'results_TWP3'

    Ours1 = r'results_Ours1'
    Ours2 = r'results_Ours2'
    Ours3 = r'results_Ours3'

    OGDLC1 = r'results_OGDLC1'
    OGDLC2 = r'results_OGDLC2'
    OGDLC3 = r'results_OGDLC3'

    Mask1 = r'results_Mask1'
    Mask2 = r'results_Mask2'
    Mask3 = r'results_Mask3'

    AdpOLC1 = r'results_AdpOLC1'
    AdpOLC2 = r'results_AdpOLC2'
    AdpOLC3 = r'results_AdpOLC13'

    mean_loss_ours, mean_dbc_ours, mean_acc_ours, mean_dp_ours, mean_eop_ours, mean_disc_ours, mean_cons_ours, std_loss_ours, std_dbc_ours, std_acc_ours, std_dp_ours, std_eop_ours, std_disc_ours, std_cons_ours = mean_std(
        Ours1, Ours2, Ours3)
    mean_loss_mask, mean_dbc_mask, mean_acc_mask, mean_dp_mask, mean_eop_mask, mean_disc_mask, mean_cons_mask, std_loss_mask, std_dbc_mask, std_acc_mask, std_dp_mask, std_eop_mask, std_disc_mask, std_cons_mask = mean_std(
        Mask1, Mask2, Mask3)
    mean_loss_twp, mean_dbc_twp, mean_acc_twp, mean_dp_twp, mean_eop_twp, mean_disc_twp, mean_cons_twp, std_loss_twp, std_dbc_twp, std_acc_twp, std_dp_twp, std_eop_twp, std_disc_twp, std_cons_twp = mean_std(
        TWP1, TWP2, TWP3)
    mean_loss_ogdlc, mean_dbc_ogdlc, mean_acc_ogdlc, mean_dp_ogdlc, mean_eop_ogdlc, mean_disc_ogdlc, mean_cons_ogdlc, std_loss_ogdlc, std_dbc_ogdlc, std_acc_ogdlc, std_dp_ogdlc, std_eop_ogdlc, std_disc_ogdlc, std_cons_ogdlc = mean_std(
        OGDLC1, OGDLC2, OGDLC3)
    mean_loss_adpolc, mean_dbc_adpolc, mean_acc_adpolc, mean_dp_adpolc, mean_eop_adpolc, mean_disc_adpolc, mean_cons_adpolc, std_loss_adpolc, std_dbc_adpolc, std_acc_adpolc, std_dp_adpolc, std_eop_adpolc, std_disc_adpolc, std_cons_adpolc = mean_std(
        AdpOLC1, AdpOLC2, AdpOLC3)
    mean_loss_genolc, mean_dbc_genolc, mean_acc_genolc, mean_dp_genolc, mean_eop_genolc, mean_disc_genolc, mean_cons_genolc, std_loss_genolc, std_dbc_genolc, std_acc_genolc, std_dp_genolc, std_eop_genolc, std_disc_genolc, std_cons_genolc = mean_std(
        GenOLC1, GenOLC2, GenOLC3)

    ##################################################
    dps = [mean_dp_ours[start_task::every], mean_dp_mask[start_task::every], mean_dp_twp[start_task::every], mean_dp_ogdlc[start_task::every], mean_dp_adpolc[start_task::every],
           mean_dp_genolc[start_task::every], mean_dp_pdfm[start_task::every]]
    dp_stds = [std_dp_ours[start_task::every], std_dp_mask[start_task::every], std_dp_twp[start_task::every], std_dp_ogdlc[start_task::every], std_dp_adpolc[start_task::every],
               std_dp_genolc[start_task::every], std_dp_pdfm[start_task::every]]
    ylim = [0, 1]
    # xlabels = range(1, num_tasks + 1)
    eval_name = 'Demographic Parity'
    plotting(dps, dp_stds, ylim, eval_name, num_tasks, dataname)
    ###########################################################################################################

    ##################################################
    eops = [mean_eop_ours[start_task::every], mean_eop_mask[start_task::every], mean_eop_twp[start_task::every], mean_eop_ogdlc[start_task::every], mean_eop_adpolc[start_task::every],
            mean_eop_genolc[start_task::every], mean_eop_pdfm[start_task::every]]
    eop_stds = [std_eop_ours[start_task::every], std_eop_mask[start_task::every], std_eop_twp[start_task::every], std_eop_ogdlc[start_task::every], std_eop_adpolc[start_task::every],
                std_eop_genolc[start_task::every], std_eop_pdfm[start_task::every]]
    ylim = [0, 1]
    eval_name = 'Equalized Odds'
    plotting(eops, eop_stds, ylim, eval_name, num_tasks, dataname)
    ###########################################################################################################
