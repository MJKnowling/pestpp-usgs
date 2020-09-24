#include <random>
#include <map>
#include <iomanip>
#include <mutex>
#include <thread>
#include "Ensemble.h"
#include "RestartController.h"
#include "utilities.h"
#include "Ensemble.h"
#include "SQP.h"
#include "ObjectiveFunc.h"
#include "covariance.h"
#include "RedSVD-h.h"
#include "SVDPackage.h"
#include "eigen_tools.h"
#include "EnsembleMethodUtils.h"



SeqQuadProgram::SeqQuadProgram(Pest &_pest_scenario, FileManager &_file_manager,
	OutputFileWriter &_output_file_writer, PerformanceLog *_performance_log,
	RunManagerAbstract* _run_mgr_ptr) : pest_scenario(_pest_scenario), file_manager(_file_manager),
	output_file_writer(_output_file_writer), performance_log(_performance_log),
	run_mgr_ptr(_run_mgr_ptr)
{
	rand_gen = std::mt19937(pest_scenario.get_pestpp_options().get_random_seed());
	subset_rand_gen = std::mt19937(pest_scenario.get_pestpp_options().get_random_seed());
	dv.set_pest_scenario(&pest_scenario);
	oe.set_pest_scenario(&pest_scenario);
	dv.set_rand_gen(&rand_gen);
	oe.set_rand_gen(&rand_gen);
	
}

void SeqQuadProgram::throw_sqp_error(string message)
{
	performance_log->log_event("SeqQuadProgram error: " + message);
	cout << endl << "   ************   " << endl << "    SeqQuadProgram error: " << message << endl << endl;
	file_manager.rec_ofstream() << endl << "   ************   " << endl << "    SeqQuadProgram error: " << message << endl << endl;
	file_manager.close_file("rec");
	performance_log->~PerformanceLog();
	throw runtime_error("SeqQuadProgram error: " + message);
}

bool SeqQuadProgram::initialize_dv(Covariance &cov)
{
	stringstream ss;
	int num_reals = pest_scenario.get_pestpp_options().get_ies_num_reals();
	string par_csv = pest_scenario.get_pestpp_options().get_ies_par_csv();
	bool drawn = false;
	if (par_csv.size() == 0)
	{
		ofstream& frec = file_manager.rec_ofstream();
		message(1, "drawing parameter realizations: ", num_reals);
		map<string, double> par_means = pest_scenario.get_ext_file_double_map("parameter data external", "mean");
		Parameters draw_par = pest_scenario.get_ctl_parameters();
		if (par_means.size() > 0)
		{
			
			frec << "Note: the following parameters contain 'mean' value information that will be used in place of " << endl;
			frec << "      the 'parval1' values as mean values during ensemble generation" << endl;
			double lb, ub;
			for (auto par_mean : par_means)
			{
				if (draw_par.find(par_mean.first) != draw_par.end())
				{
					lb = pest_scenario.get_ctl_parameter_info().get_parameter_rec_ptr(par_mean.first)->lbnd;
					ub = pest_scenario.get_ctl_parameter_info().get_parameter_rec_ptr(par_mean.first)->ubnd;
					if (par_mean.second < lb)
					{
						frec << "Warning: 'mean' value for parameter " << par_mean.first << " less than lower bound, using 'parval1'";
					}
					else if (par_mean.second > ub)
					{
						frec << "Warning: 'mean' value for parameter " << par_mean.first << " greater than upper bound, using 'parval1'";
					}
					else
					{
						draw_par[par_mean.first] = par_mean.second;
						frec << par_mean.first << " " << par_mean.second << endl;
					}
					
				}

			}
		}
		dv.draw(num_reals, draw_par,cov, performance_log, pest_scenario.get_pestpp_options().get_ies_verbose_level(), file_manager.rec_ofstream());
		// stringstream ss;
		// ss << file_manager.get_base_filename() << ".0.par.csv";
		// message(1, "saving initial parameter ensemble to ", ss.str());
		// dv.to_csv(ss.str());
		drawn = true;
	}
	else
	{
		string par_ext = pest_utils::lower_cp(par_csv).substr(par_csv.size() - 3, par_csv.size());
		performance_log->log_event("processing par csv " + par_csv);
		if (par_ext.compare("csv") == 0)
		{
			message(1, "loading par ensemble from csv file", par_csv);
			try
			{
				dv.from_csv(par_csv);
			}
			catch (const exception &e)
			{
				ss << "error processing par csv: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing par csv"));
			}
		}
		else if ((par_ext.compare("jcb") == 0) || (par_ext.compare("jco") == 0))
		{
			message(1, "loading par ensemble from binary file", par_csv);
			try
			{
				dv.from_binary(par_csv);
			}
			catch (const exception &e)
			{
				ss << "error processing par jcb: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing par jcb"));
			}
		}
		else
		{
			ss << "unrecognized par csv extension " << par_ext << ", looking for csv, jcb, or jco";
			throw_sqp_error(ss.str());
		}

		dv.transform_ip(ParameterEnsemble::transStatus::NUM);
		
		if (pp_args.find("IES_NUM_REALS") != pp_args.end())
		{
			int num_reals = pest_scenario.get_pestpp_options().get_ies_num_reals();
			/*if (pest_scenario.get_pestpp_options().get_ies_include_base())
			{
				message(1, "Note: increasing num_reals by 1 to account for 'base' realization in existing par ensemble");
				num_reals++;
			}*/
			if (num_reals < dv.shape().first)
			{
				message(1,"ies_num_reals arg passed, truncated parameter ensemble to ",num_reals);
				vector<string> keep_names,real_names=dv.get_real_names();
				for (int i=0;i<num_reals;i++)
				{
					keep_names.push_back(real_names[i]);
				}
				dv.keep_rows(keep_names);
			}
		}
		

		if (pest_scenario.get_pestpp_options().get_ies_use_empirical_prior())
		{
			message(1, "initializing prior parameter covariance matrix from ensemble (using diagonal matrix)");
			parcov = dv.get_diagonal_cov_matrix();
			if (pest_scenario.get_pestpp_options().get_ies_verbose_level() > 1)
			{

				if (pest_scenario.get_pestpp_options().get_ies_save_binary())
				{
					string filename = file_manager.get_base_filename() + ".prior.jcb";
					message(1, "saving emprirical parameter covariance matrix to binary file: ", filename);
					parcov.to_binary(filename);
				}
				else
				{
					string filename = file_manager.get_base_filename() + ".prior.cov";
					message(1, "saving emprirical parameter covariance matrix to ASCII file: ", filename);
					parcov.to_ascii(filename);
				}
			}
		}
		if (pest_scenario.get_pestpp_options().get_ies_enforce_bounds())
		{
			if (pest_scenario.get_pestpp_options().get_ies_obs_restart_csv().size() > 0)
				message(1, "Warning: even though ies_enforce_bounds is true, a restart obs en was passed, so bounds will not be enforced on the initial par en");
			else
				dv.enforce_limits(performance_log, pest_scenario.get_pestpp_options().get_ies_enforce_chglim());
		}

	}
	return drawn;

}

void SeqQuadProgram::add_bases()
{
	//check that 'base' isn't already in ensemble
	vector<string> rnames = dv.get_real_names();
	bool inpar = false;
	if (find(rnames.begin(), rnames.end(), base_name) != rnames.end())
	{
		message(1, "'base' realization already in parameter ensemble, ignoring '++ies_include_base'");
		inpar = true;
	}
	else
	{
		message(1, "adding 'base' parameter values to ensemble");
		Parameters pars = pest_scenario.get_ctl_parameters();
		dv.get_par_transform().active_ctl2numeric_ip(pars);
		vector<int> drop{ dv.shape().first - 1 };
		dv.drop_rows(drop);
		dv.append(base_name, pars);
	}

	//check that 'base' isn't already in ensemble
	rnames = oe.get_real_names();
	if (find(rnames.begin(), rnames.end(), base_name) != rnames.end())
	{
		message(1, "'base' realization already in observation ensemble, ignoring '++ies_include_base'");
	}
	else
	{
		Observations obs = pest_scenario.get_ctl_observations();
		if (inpar)
		{
			vector<string> prnames = dv.get_real_names();

			int idx = find(prnames.begin(), prnames.end(), base_name) - prnames.begin();
			//cout << idx << "," << rnames.size() << endl;
			string oreal = rnames[idx];
			stringstream ss;
			ss << "warning: 'base' realization in par ensenmble but not in obs ensemble," << endl;
			ss << "         replacing obs realization '" << oreal << "' with 'base'";
			string mess = ss.str();
			message(1, mess);
			vector<string> drop;
			drop.push_back(oreal);
			oe.drop_rows(drop);
			oe.append(base_name, obs);
			//rnames.insert(rnames.begin() + idx, string(base_name));
			rnames[idx] = base_name;
			oe.reorder(rnames, vector<string>());
		}
		else
		{
			message(1, "adding 'base' observation values to ensemble");
			vector<int> drop{ oe.shape().first - 1 };
			oe.drop_rows(drop);
			oe.append(base_name, obs);
		}
	}
}

bool SeqQuadProgram::initialize_oe(Covariance &cov)
{
	stringstream ss;
	int num_reals = dv.shape().first;
	string obs_csv = pest_scenario.get_pestpp_options().get_ies_obs_csv();
	bool drawn = false;
	if (obs_csv.size() == 0)
	{
		if (pest_scenario.get_pestpp_options().get_ies_no_noise())
		{
			message(1, "initializing no-noise observation ensemble of : ", num_reals);
			oe.initialize_without_noise(num_reals);
		
		}
		else
		{
			message(1, "drawing observation noise realizations: ", num_reals);
			oe.draw(num_reals, cov, performance_log, pest_scenario.get_pestpp_options().get_ies_verbose_level(), file_manager.rec_ofstream());
			
		}
		drawn = true;
	}
	else
	{
		string obs_ext = pest_utils::lower_cp(obs_csv).substr(obs_csv.size() - 3, obs_csv.size());
		performance_log->log_event("processing obs csv " + obs_csv);
		if (obs_ext.compare("csv") == 0)
		{
			message(1, "loading obs ensemble from csv file", obs_csv);
			try
			{
				oe.from_csv(obs_csv);
			}
			catch (const exception &e)
			{
				ss << "error processing obs csv: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing obs csv"));
			}
		}
		else if ((obs_ext.compare("jcb") == 0) || (obs_ext.compare("jco") == 0))
		{
			message(1, "loading obs ensemble from binary file", obs_csv);
			try
			{
				oe.from_binary(obs_csv);
			}
			catch (const exception &e)
			{
				stringstream ss;
				ss << "error processing obs binary file: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing obs binary file"));
			}
		}
		else
		{
			ss << "unrecognized obs ensemble extension " << obs_ext << ", looking for csv, jcb, or jco";
			throw_sqp_error(ss.str());
		}
		if (pp_args.find("IES_NUM_REALS") != pp_args.end())
		{
			int num_reals = pest_scenario.get_pestpp_options().get_ies_num_reals();
			/*if (pest_scenario.get_pestpp_options().get_ies_include_base())
			{
				message(1, "Note: increasing num_reals by 1 to account for 'base' realization in existing obs ensemble");
				num_reals++;
			}*/
			if (num_reals < oe.shape().first)
			{
				message(1,"ies_num_reals arg passed, truncated observation ensemble to ",num_reals);
				vector<string> keep_names,real_names=oe.get_real_names();
				for (int i=0;i<num_reals;i++)
				{
					keep_names.push_back(real_names[i]);
				}
				oe.keep_rows(keep_names);
			}
		}
	}
	return drawn;

}

//template<typename T, typename A>
//void SeqQuadProgram::message(int level, char* _message, vector<T, A> _extras)
//{
//	string s(_message);
//	message(level, s, _extras);
//}

//void SeqQuadProgram::message(int level, char* _message)
//{
//	string s(_message);
//	message(level, s);
//}

//template<typename T>
//void SeqQuadProgram::message(int level, char* _message, T extra)
//{
//	string s(_message);
//	message(level, s, extra);
//
//}

template<typename T, typename A>
void SeqQuadProgram::message(int level, const string &_message, vector<T, A> _extras, bool echo)
{
	stringstream ss;
	if (level == 0)
		ss << endl << "  ---  ";
	else if (level == 1)
		ss << "...";
	ss << _message;
	if (_extras.size() > 0)
	{

		for (auto &e : _extras)
			ss << e << " , ";

	}
	if (level == 0)
		ss << "  ---  ";
	if ((echo) && ((verbose_level >= 2) || (level < 2)))
		cout << ss.str() << endl;
	file_manager.rec_ofstream() <<ss.str() << endl;
	performance_log->log_event(_message);

}

void SeqQuadProgram::message(int level, const string &_message)
{
	message(level, _message, vector<string>());
}

template<typename T>
void SeqQuadProgram::message(int level, const string &_message, T extra)
{
	stringstream ss;
	ss << _message << " " << extra;
	string s = ss.str();
	message(level, s);
}

void SeqQuadProgram::sanity_checks()
{
	PestppOptions* ppo = pest_scenario.get_pestpp_options_ptr();
	vector<string> errors;
	vector<string> warnings;
	stringstream ss;
	string par_csv = ppo->get_ies_par_csv();
	string obs_csv = ppo->get_ies_obs_csv();
	string restart_obs = ppo->get_ies_obs_restart_csv();
	string restart_par = ppo->get_ies_par_restart_csv();


	if (pest_scenario.get_control_info().pestmode == ControlInfo::PestMode::REGUL)
	{
		warnings.push_back("'pestmode' == 'regularization', in pestpp-ies, this is controlled with the ++ies_reg_factor argument, resetting to 'estimation'");
		//throw_sqp_error("'pestmode' == 'regularization', please reset to 'estimation'");
	}
	else if (pest_scenario.get_control_info().pestmode == ControlInfo::PestMode::UNKNOWN)
	{
		warnings.push_back("unrecognized 'pestmode', using 'estimation'");
	}


	if (pest_scenario.get_ctl_ordered_pi_names().size() > 0)
	{
		warnings.push_back("prior information equations not supported in pestpp-ies, ignoring...");
	}
	double acc_phi = ppo->get_ies_accept_phi_fac();
	if (acc_phi < 1.0)
		warnings.push_back("ies_accept_phi_fac < 1.0, not good!");
	if (acc_phi > 10.0)
		warnings.push_back("ies_accept_phi_fac > 10.0, this is prob too big, typical values 1.05 to 1.3");

	double lam_inc = ppo->get_ies_lambda_inc_fac();
	if (lam_inc < 1.0)
		errors.push_back("ies_lambda_inc_fac < 1.0, nope! how can lambda increase if this is less than 1.0???");

	double lam_dec = ppo->get_ies_lambda_dec_fac();
	if (lam_dec > 1.0)
		errors.push_back("ies_lambda_dec_fac > 1.0, nope!  how can lambda decrease if this is greater than 1.0???");

	if ((ppo->get_ies_par_csv().size() == 0) && (ppo->get_ies_use_empirical_prior()))
	{
		warnings.push_back("no point in using an empirical prior if we are drawing the par ensemble...resetting ies_use_empirical_prior to false");
		ppo->set_ies_use_empirical_prior(false);
	}
	if ((par_csv.size() == 0) && (restart_par.size() > 0))
		errors.push_back("ies_par_en is empty but ies_restart_par_en is not - how can this work?");
	if ((restart_par.size() > 0) && (restart_obs.size() == 0))
		errors.push_back("use of ies_restart_par_en requires ies_restart_obs_en");
	if ((par_csv.size() == 0) && (restart_obs.size() > 0))
		errors.push_back("ies_par_en is empty but ies_restart_obs_en is not - how can this work?");
	if (ppo->get_ies_bad_phi() <= 0.0)
		errors.push_back("ies_bad_phi <= 0.0, really?");
	if ((ppo->get_ies_num_reals() < error_min_reals) && (par_csv.size() == 0))
	{
		ss.str("");
		ss << "ies_num_reals < " << error_min_reals << ", this is redic, increaing to " << warn_min_reals;
		warnings.push_back(ss.str());
		ppo->set_ies_num_reals(warn_min_reals);
	}
	if ((ppo->get_ies_num_reals() < warn_min_reals) && (par_csv.size() == 0))
	{
		ss.str("");
		ss << "ies_num_reals < " << warn_min_reals << ", this is prob too few";
		warnings.push_back(ss.str());
	}
	if (ppo->get_ies_reg_factor() < 0.0)
		errors.push_back("ies_reg_factor < 0.0 - WRONG!");
	//if (ppo->get_ies_reg_factor() > 1.0)
	//	errors.push_back("ies_reg_factor > 1.0 - nope");
	if ((par_csv.size() == 0) && (ppo->get_ies_subset_size() < 10000000) && (ppo->get_ies_num_reals() < ppo->get_ies_subset_size() * 2))
		warnings.push_back("ies_num_reals < 2*ies_subset_size: you not gaining that much using subset here");
	//if ((ppo->get_ies_subset_size() < 100000001) && (ppo->get_ies_lam_mults().size() == 1))
	//{
	//	warnings.push_back("only one lambda mult to test, no point in using a subset");
	//	//ppo->set_ies_subset_size(100000000);
	//}

	string how = pest_scenario.get_pestpp_options().get_ies_subset_how();
	if ((how != "FIRST") && (how != "LAST") && (how != "RANDOM") && (how != "PHI_BASED"))
	{
		ss.str("");
		ss << "'subset_how' is '" << how << "' but should be 'FIRST','LAST','RANDOM','PHI_BASED'";
		errors.push_back(ss.str());
	}

	if ((ppo->get_ies_verbose_level() < 0) || (ppo->get_ies_verbose_level() > 3))
	{
		warnings.push_back("ies_verbose_level must be between 0 and 3, resetting to 3");
		ppo->set_ies_verbose_level(3);
	}
	if ((ppo->get_ies_no_noise()) && (ppo->get_obscov_filename().size() > 0))
	{
		ss.str("");
		ss << "ies_no_noise is true but obscov file supplied - these two are not compatible";
		errors.push_back(ss.str());
	}

	if ((ppo->get_obscov_filename().size() > 0) && (ppo->get_ies_drop_conflicts()))
	{
		ss.str("");
		ss << "use of a full obscov with ies_drop_conflicts is not currently supported";
		errors.push_back(ss.str());
	}
	if ((ppo->get_ies_obs_csv().size() > 0) && (ppo->get_ies_no_noise()))
	{
		ss.str("");
		ss << "ies_no_noise can't be used with an ies_observation_ensemble";
		errors.push_back(ss.str());
	}
	
	/*if (!ppo->get_ies_use_prior_scaling())
	{
		warnings.push_back("not using prior scaling - this is really a dev option, you should always use prior scaling...");
	}*/

	/*if ((ppo->get_ies_num_threads() > 0) && (!use_localizer))
	{
		warnings.push_back("'ies_num_threads > 0 but no localization, resetting 'ies_num_threads' = 0");
		ppo->set_ies_num_threads(-1);
		num_threads = -1;
	}
*/


	if (warnings.size() > 0)
	{
		message(0, "sanity_check warnings");
		for (auto &w : warnings)
			message(1, w);
		message(1, "continuing initialization...");
	}
	if (errors.size() > 0)
	{
		message(0, "sanity_check errors - uh oh");
		for (auto &e : errors)
			message(1, e);
		throw_sqp_error(string("sanity_check() found some problems - please review rec file"));
	}
	//cout << endl << endl;
}

void SeqQuadProgram::initialize_restart()
{
	stringstream ss;
	string obs_restart_csv = pest_scenario.get_pestpp_options().get_ies_obs_restart_csv();
	string par_restart_csv = pest_scenario.get_pestpp_options().get_ies_par_restart_csv();

	//performance_log->log_event("restart with existing obs ensemble: " + obs_restart_csv);
	message(1, "restarting with existing obs ensemble", obs_restart_csv);
	string obs_ext = pest_utils::lower_cp(obs_restart_csv).substr(obs_restart_csv.size() - 3, obs_restart_csv.size());
	if (obs_ext.compare("csv") == 0)
	{
		message(1, "loading restart obs ensemble from csv file", obs_restart_csv);
		try
		{
			oe.from_csv(obs_restart_csv);
		}
		catch (const exception &e)
		{
			ss << "error processing restart obs csv: " << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("error processing restart obs csv"));
		}
	}
	else if ((obs_ext.compare("jcb") == 0) || (obs_ext.compare("jco") == 0))
	{
		message(1, "loading restart obs ensemble from binary file", obs_restart_csv);
		try
		{
			oe.from_binary(obs_restart_csv);
		}
		catch (const exception &e)
		{
			ss << "error processing restart obs binary file: " << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("error processing restart obs binary file"));
		}
	}
	else
	{
		ss << "unrecognized restart obs ensemble extension " << obs_ext << ", looking for csv, jcb, or jco";
		throw_sqp_error(ss.str());
	}
	if (par_restart_csv.size() > 0)
	{
		string par_ext = pest_utils::lower_cp(par_restart_csv).substr(par_restart_csv.size() - 3, par_restart_csv.size());
		if (par_ext.compare("csv") == 0)
		{
			message(1, "loading restart par ensemble from csv file", par_restart_csv);
			try
			{
				dv.from_csv(par_restart_csv);
			}
			catch (const exception &e)
			{
				ss << "error processing restart par csv: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing restart par csv"));
			}
		}
		else if ((par_ext.compare("jcb") == 0) || (par_ext.compare("jco") == 0))
		{
			message(1, "loading restart par ensemble from binary file", par_restart_csv);
			try
			{
				dv.from_binary(par_restart_csv);
			}
			catch (const exception &e)
			{
				ss << "error processing restart par binary file: " << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error processing restart par binary file"));
			}
		}
		else
		{
			ss << "unrecognized restart par ensemble extension " << par_ext << ", looking for csv, jcb, or jco";
			throw_sqp_error(ss.str());
		}
		if (dv.shape().first != oe.shape().first)
		{
			ss.str("");
			ss << "restart par en has " << dv.shape().first << " realizations but restart obs en has " << oe.shape().first;
			throw_sqp_error(ss.str());
		}

		//check that restart dv is in sync with dv_base
		vector<string> pe_real_names = dv.get_real_names(), pe_base_real_names = dv_base.get_real_names();
		vector<string>::const_iterator start, end;
		vector<string> missing;
		start = pe_base_real_names.begin();
		end = pe_base_real_names.end();
		for (auto &rname : pe_real_names)
			if (find(start, end, rname) == end)
				missing.push_back(rname);
		if (missing.size() > 0)
		{
			ss << "the following realization names were found in the restart par en but not in the 'base' par en:";
			for (auto &m : missing)
				ss << m << ",";
			throw_sqp_error(ss.str());
		}
		dv.transform_ip(ParameterEnsemble::transStatus::NUM);
	}

	if (pp_args.find("IES_NUM_REALS") != pp_args.end())
	{
		int num_reals = pest_scenario.get_pestpp_options().get_ies_num_reals();
		/*if (pest_scenario.get_pestpp_options().get_ies_include_base())
		{
			message(1, "Note: increasing num_reals by 1 to account for 'base' realization in existing obs restart ensemble");
			num_reals++;
		}*/
		if (num_reals < oe.shape().first)
		{
			message(1, "ies_num_reals arg passed, truncated restart obs ensemble to ", num_reals);
			vector<string> keep_names, real_names = oe.get_real_names();
			for (int i = 0; i<num_reals; i++)
			{
				keep_names.push_back(real_names[i]);
			}
			oe.keep_rows(keep_names);
		}
	}

	//check that restart oe is in sync with oe_base
	vector<string> oe_real_names = oe.get_real_names(), oe_base_real_names = oe_base.get_real_names();
	vector<string>::const_iterator start, end;
	vector<string> missing;
	start = oe_base_real_names.begin();
	end = oe_base_real_names.end();
	for (auto &rname : oe_real_names)
		if (find(start, end, rname) == end)
			missing.push_back(rname);
	if (missing.size() > 0)
	{
		//the special case where the base real is what is missing...
		if ((missing.size() == 1) && (missing[0] == "BASE"))
		{
			//check that the base real is in the par en - restart_par_en should be accounted for by now
			int base_par_idx = -1;
			vector<string> pe_real_names = dv.get_real_names(), pe_base_real_names = dv_base.get_real_names();
			for (int i = 0; i < dv_base.shape().first; i++)
			{
				if (pe_base_real_names[i] == "BASE")
				{
					base_par_idx = i;
					break;
				}
			}
			if (base_par_idx != -1)
			{
				ss.str("");
				ss << "WARNING: replacing base obs en realization '" << oe_base_real_names[base_par_idx] << "' with 'base' (noise free) values to match par en 'base' location";
				message(2, ss.str());
				Observations obs = pest_scenario.get_ctl_observations();
				oe_base.replace(base_par_idx, obs, "BASE");
			}
			else
			{
				ss << "the 'base' realization was not found in the restart obs en and also not found in the par en";
				throw_sqp_error(ss.str());
			}

		}
		else
		{
			ss << "the following realization names were found in the restart obs en but not in the 'base' obs en:";
			for (auto &m : missing)
				ss << m << ",";
			throw_sqp_error(ss.str());
		}

	}

	if (oe.shape().first != dv.shape().first)
	{
		//check if all oe names are found in par en, if so, we can reorder and proceed.  otherwise, die
		missing.clear();
		vector<string> pe_real_names = dv.get_real_names();
		for (auto &oname : oe_real_names)
		{
			if (find(pe_real_names.begin(), pe_real_names.end(), oname) == pe_real_names.end())
				missing.push_back(oname);
		}

		if (missing.size() > 0)
		{
			ss << "number of reals differ between restart obs en (" << oe.shape().first << ") and par en (" << dv.shape().first << ")";
			ss << " and realization names could not be aligned:";
			for (auto &m : missing)
				ss << m << ",";
			throw_sqp_error(ss.str());
		}

		message(2, "reordering dv to align with restart obs en, num reals: ", oe_real_names.size());
		try
		{
			dv.reorder(oe_real_names, vector<string>());
		}
		catch (exception &e)
		{
			ss << "error reordering dv with restart oe:" << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("error reordering dv with restart oe"));
		}

	}

	//if (oe.shape().first < oe_base.shape().first) //maybe some runs failed...
	if (oe.shape().first <= oe_base.shape().first)
	{
		//find which realizations are missing and reorder oe_base, dv and dv_base

		//message(1, "shape mismatch detected with restart obs ensemble...checking for compatibility");
		
		/*vector<string> pe_real_names;
		start = oe_base_real_names.begin();
		end = oe_base_real_names.end();
		vector<string>::const_iterator it;
		int iit;
		for (int i = 0; i < oe.shape().first; i++)
		{
			it = find(start, end, oe_real_names[i]);
			if (it != end)
			{
				iit = it - start;
				pe_real_names.push_back(pe_org_real_names[iit]);
			}
		}*/
		message(2, "reordering oe_base to align with restart obs en,num reals:", oe_real_names.size());
		if ((oe_drawn) && (oe_base.shape().first == oe_real_names.size()))
		{
			oe_base.set_real_names(oe_real_names);
		}
		else
		{
			try
			{
				oe_base.reorder(oe_real_names, vector<string>());
			}
			catch (exception& e)
			{
				ss << "error reordering oe_base with restart oe:" << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error reordering oe_base with restart oe"));
			}
		}
		//if (par_restart_csv.size() > 0)
		if (true)
		{
			vector<string> pe_real_names = dv.get_real_names();
			message(2, "reordering dv_base to align with restart par en,num reals:", pe_real_names.size());
			try
			{
				dv_base.reorder(pe_real_names, vector<string>());
			}
			catch (exception &e)
			{
				ss << "error reordering dv_base with restart dv:" << e.what();
				throw_sqp_error(ss.str());
			}
			catch (...)
			{
				throw_sqp_error(string("error reordering dv_base with restart dv"));
			}
		}


		/*try
		{
			dv.reorder(pe_real_names, vector<string>());
		}
		catch (exception &e)
		{
			ss << "error reordering dv with restart oe:" << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("error reordering dv with restart oe"));
		}*/
	}
	else if (oe.shape().first > oe_base.shape().first) //something is wrong
	{
		ss << "restart oe has too many rows: " << oe.shape().first << " compared to oe_base: " << oe_base.shape().first;
		throw_sqp_error(ss.str());
	}
}


void SeqQuadProgram::initialize_parcov()
{
	stringstream ss;
	performance_log->log_event("initializing parcov");

	if (pest_scenario.get_pestpp_options().get_ies_use_empirical_prior())
		return;
	string how = parcov.try_from(pest_scenario, file_manager);
	message(1, "parcov loaded ", how);
	//if (parcov.e_ptr()->rows() > 0)
	parcov = parcov.get(act_par_names);

}


void SeqQuadProgram::initialize_obscov()
{
	message(1, "initializing observation noise covariance matrix");
	string obscov_filename = pest_scenario.get_pestpp_options().get_obscov_filename();

	string how = obscov.try_from(pest_scenario, file_manager, false, true);
	message(1, "obscov loaded ", how);
	
}


void SeqQuadProgram::initialize()
{	
	message(0, "initializing");
	pp_args = pest_scenario.get_pestpp_options().get_passed_args();

	act_obs_names = pest_scenario.get_ctl_ordered_nz_obs_names();
	act_par_names = pest_scenario.get_ctl_ordered_adj_par_names();

	stringstream ss;

	if (pest_scenario.get_control_info().noptmax == 0)
	{
		message(0, "'noptmax'=0, running control file parameter values and quitting");
		
		Parameters pars = pest_scenario.get_ctl_parameters();
		ParamTransformSeq pts = dv.get_par_transform();

		ParameterEnsemble _pe(&pest_scenario, &rand_gen);
		_pe.reserve(vector<string>(), pest_scenario.get_ctl_ordered_par_names());
		_pe.set_trans_status(ParameterEnsemble::transStatus::CTL);
		_pe.append("BASE", pars);
		string par_csv = file_manager.get_base_filename() + ".par.csv";
		//message(1, "saving parameter values to ", par_csv);
		//_pe.to_csv(par_csv);
		dv_base = _pe;
		dv_base.reorder(vector<string>(), act_par_names);
		ObservationEnsemble _oe(&pest_scenario, &rand_gen);
		_oe.reserve(vector<string>(), pest_scenario.get_ctl_ordered_obs_names());
		_oe.append("BASE", pest_scenario.get_ctl_observations());
		oe_base = _oe;
		oe_base.reorder(vector<string>(), act_obs_names);
		//initialize the phi handler
		ph = L2PhiHandler(&pest_scenario, &file_manager, &oe_base, &dv_base, &parcov);
		if (ph.get_lt_obs_names().size() > 0)
		{
			message(1, "less_than inequality defined for observations: ", ph.get_lt_obs_names().size());
		}
		if (ph.get_gt_obs_names().size())
		{
			message(1, "greater_than inequality defined for observations: ", ph.get_gt_obs_names().size());
		}
		message(1, "running control file parameter values");

		vector<int> failed_idxs = run_ensemble(_pe, _oe);
		if (failed_idxs.size() != 0)
		{
			message(0, "control file parameter value run failed...bummer");
			throw_sqp_error("control file parameter value run failed");
		}
		string obs_csv = file_manager.get_base_filename() + ".obs.csv";
		message(1, "saving results from control file parameter value run to ", obs_csv);
		_oe.to_csv(obs_csv);

		ph.update(_oe, _pe);
		message(0, "control file parameter phi report:");
		ph.report(true);
		ph.write(0, 1);
		save_base_real_par_rei(pest_scenario, _pe, _oe, output_file_writer, file_manager, -1);			
		return;
	}

	//set some defaults
	PestppOptions *ppo = pest_scenario.get_pestpp_options_ptr();

	
	iter = 0;
	//ofstream &frec = file_manager.rec_ofstream();
	last_best_mean = 1.0E+30;
	last_best_std = 1.0e+30;
	
	warn_min_reals = 10;
	error_min_reals = 2;
	
	vector<double> scale_facs = pest_scenario.get_pestpp_options().get_lambda_scale_vec();
	message(1, "using lambda scaling factors: ", scale_facs);
	double acc_fac = pest_scenario.get_pestpp_options().get_ies_accept_phi_fac();
	message(1, "acceptable phi factor: ", acc_fac);
	double inc_fac = pest_scenario.get_pestpp_options().get_ies_lambda_inc_fac();
	message(1, "lambda increase factor: ", inc_fac);
	double dec_fac = pest_scenario.get_pestpp_options().get_ies_lambda_dec_fac();
	message(1, "lambda decrease factor: ", dec_fac);
	message(1, "max run fail: ", ppo->get_max_run_fail());

	sanity_checks();

	bool echo = false;
	if (verbose_level > 1)
		echo = true;

	initialize_parcov();
	initialize_obscov();

	subset_size = pest_scenario.get_pestpp_options().get_ies_subset_size();
	reg_factor = pest_scenario.get_pestpp_options().get_ies_reg_factor();
	message(1,"using reg_factor: ", reg_factor);
	double bad_phi = pest_scenario.get_pestpp_options().get_ies_bad_phi();
	if (bad_phi < 1.0e+30)
		message(1, "using bad_phi: ", bad_phi);

	int num_reals = pest_scenario.get_pestpp_options().get_ies_num_reals();

	dv_drawn = initialize_dv(parcov);

	oe_drawn = initialize_oe(obscov);
	string center_on = ppo->get_ies_center_on();
	
	try
	{
		dv.check_for_dups();
	}
	catch (const exception &e)
	{
		string message = e.what();
		throw_sqp_error("error in parameter ensemble: " + message);
	}

	try
	{
		oe.check_for_dups();
	}
	catch (const exception &e)
	{
		string message = e.what();
		throw_sqp_error("error in observation ensemble: " + message);
	}

	if (dv.shape().first != oe.shape().first)
	{
		//the special case where par en < obs en and all par reals are found in obs en...

		if (dv.shape().first < oe.shape().first)
		{
			vector<string> oe_names = oe.get_real_names();
			set<string> oset(oe_names.begin(), oe_names.end());
			vector<string> missing;
			for (auto n : dv.get_real_names())
				if (oset.find(n) == oset.end())
				{
					missing.push_back(n);
				}
			if (missing.size() == 0)
			{
				ss.str("");
				ss << "par en has " << dv.shape().first << " realizations, compared to " << oe.shape().first << " obs realizations";
				message(1, ss.str());
				message(1," the realization names are compatible");
				message(1, "re-indexing obs en to align with par en...");

				oe.reorder(dv.get_real_names(), vector<string>());
			}
			else
			{
				ss.str("");
				ss << "the following par en real names were not found in the obs en: ";
				for (auto m : missing)
				{
					ss << m << ",";
				}
				throw_sqp_error(ss.str());
			}
		}
		else
		{
			ss.str("");
			ss << "parameter ensemble rows (" << dv.shape().first << ") not equal to observation ensemble rows (" << oe.shape().first << ")";
			throw_sqp_error(ss.str());
		}
	}


	//need this here for Am calcs...
	//message(1, "transforming parameter ensemble to numeric");
	dv.transform_ip(ParameterEnsemble::transStatus::NUM);

	if (pest_scenario.get_pestpp_options().get_ies_include_base())
		if (pp_args.find("IES_RESTART_OBS_EN") != pp_args.end())
		{
			message(1, "Warning: even though `ies_include_base` is true, you passed a restart obs en, not adding 'base' realization...");
		}
		else
			add_bases();


	//now we check to see if we need to try to align the par and obs en
	//this would only be needed if either of these were not drawn
	if (!dv_drawn || !oe_drawn)
	{
		bool aligned = dv.try_align_other_rows(performance_log, oe);
		if (aligned)
		{
			message(2, "observation ensemble reordered to align rows with parameter ensemble");
		}
	}

	//just check to see if common real names are found but are not in the same location
	map<string, int> pe_map = dv.get_real_map(), oe_map = oe.get_real_map();
	vector<string> misaligned;
	for (auto item : pe_map)
	{
		if (oe_map.find(item.first) == oe_map.end())
			continue;
		if (item.second != oe_map[item.first])
			misaligned.push_back(item.first);
	}
	if (misaligned.size() > 0)
	{
		message(1, "WARNING: common realization names shared between the parameter and observation ensembles but they are not in the same row locations, see .rec file for listing");
		ofstream& frec = file_manager.rec_ofstream();
		frec << endl <<  "WARNING: the following " << misaligned.size() << " realization names are shared between the parameter and observation ensembles but they are not in the same row locations:" << endl;
		for (auto ma : misaligned)
			frec << ma << endl;
	}



	message(2, "checking for denormal values in dv");
	dv.check_for_normal("initial transformed parameter ensemble");
	ss.str("");
	if (pest_scenario.get_pestpp_options().get_ies_save_binary())
	{
		ss << file_manager.get_base_filename() << ".0.par.jcb";
		dv.to_binary(ss.str());
	}
	else
	{
		ss << file_manager.get_base_filename() << ".0.par.csv";
		dv.to_csv(ss.str());
	}
	message(1, "saved initial parameter ensemble to ", ss.str());
	message(2, "checking for denormal values in base oe");
	oe.check_for_normal("obs+noise observation ensemble");
	ss.str("");
	if (pest_scenario.get_pestpp_options().get_ies_save_binary())
	{
		ss << file_manager.get_base_filename() << ".obs+noise.jcb";
		oe.to_binary(ss.str());
	}
	else
	{
		ss << file_manager.get_base_filename() << ".obs+noise.csv";
		oe.to_csv(ss.str());
	}
	message(1, "saved obs+noise observation ensemble (obsval+noise) to ", ss.str());


	if (center_on.size() > 0)
	{
		ss.str("");
		ss << "centering on realization: '" << center_on << "' ";
		message(1, ss.str());
		vector<string> names = dv.get_real_names();
		if (find(names.begin(), names.end(), center_on) == names.end())
			throw_sqp_error("'ies_center_on' realization not found in par en: " + center_on);
		names = oe.get_real_names();
		if (find(names.begin(), names.end(), center_on) == names.end())
			throw_sqp_error("'ies_center_on' realization not found in obs en: " + center_on);
	}
	else
		message(1, "centering on ensemble mean vector");

	if (pest_scenario.get_control_info().noptmax == -2)
	{
		message(0, "'noptmax'=-2, running mean parameter ensemble values and quitting");
		message(1, "calculating mean parameter values");
		Parameters pars;
		vector<double> mv = dv.get_mean_stl_vector();
		pars.update(dv.get_var_names(), dv.get_mean_stl_vector());
		ParamTransformSeq pts = dv.get_par_transform();

		ParameterEnsemble _pe(&pest_scenario, &rand_gen);
		_pe.reserve(vector<string>(), dv.get_var_names());
		_pe.set_trans_status(dv.get_trans_status());
		_pe.append("mean", pars);
		string par_csv = file_manager.get_base_filename() + ".mean.par.csv";
		message(1, "saving mean parameter values to ", par_csv);
		_pe.to_csv(par_csv);
		dv_base = _pe;
		dv_base.reorder(vector<string>(), act_par_names);
		ObservationEnsemble _oe(&pest_scenario, &rand_gen);
		_oe.reserve(vector<string>(), oe.get_var_names());
		_oe.append("mean", pest_scenario.get_ctl_observations());
		oe_base = _oe;
		oe_base.reorder(vector<string>(), act_obs_names);
		//initialize the phi handler
		ph = L2PhiHandler(&pest_scenario, &file_manager, &oe_base, &dv_base, &parcov);
		if (ph.get_lt_obs_names().size() > 0)
		{
			message(1, "less_than inequality defined for observations: ", ph.get_lt_obs_names().size());
		}
		if (ph.get_gt_obs_names().size())
		{
			message(1, "greater_than inequality defined for observations: ", ph.get_gt_obs_names().size());
		}
		message(1, "running mean parameter values");

		vector<int> failed_idxs = run_ensemble(_pe, _oe);
		if (failed_idxs.size() != 0)
		{
			message(0, "mean parmeter value run failed...bummer");
			return;
		}
		string obs_csv = file_manager.get_base_filename() + ".mean.obs.csv";
		message(1, "saving results from mean parameter value run to ", obs_csv);
		_oe.to_csv(obs_csv);

		ph.update(_oe, _pe);
		message(0, "mean parameter phi report:");
		ph.report(true);
		ph.write(0, 1);

		return;
	}

	if (subset_size > dv.shape().first)
	{
		use_subset = false;
	}
	else
	{
		message(1, "using subset in lambda testing, number of realizations used in subset testing: ", subset_size);
		string how = pest_scenario.get_pestpp_options().get_ies_subset_how();
		message(1, "subset how: ", how);
		use_subset = true;
	}

	oe_org_real_names = oe.get_real_names();
	pe_org_real_names = dv.get_real_names();
	string obs_restart_csv = pest_scenario.get_pestpp_options().get_ies_obs_restart_csv();
	string par_restart_csv = pest_scenario.get_pestpp_options().get_ies_par_restart_csv();

	oe_base = oe; //copy
	//reorder this for later...
	oe_base.reorder(vector<string>(), act_obs_names);

	dv_base = dv; //copy
	//reorder this for later
	dv_base.reorder(vector<string>(), act_par_names);


	//the hard way to restart
	if (obs_restart_csv.size() > 0)
		initialize_restart();
	
	//no restart
	else
	{
		performance_log->log_event("running initial ensemble");
		message(1, "running initial ensemble of size", oe.shape().first);
		vector<int> failed = run_ensemble(dv, oe);
		if (dv.shape().first == 0)
			throw_sqp_error("all realizations failed during initial evaluation");

		dv.transform_ip(ParameterEnsemble::transStatus::NUM);
	}
	
	ss.str("");
	if (pest_scenario.get_pestpp_options().get_ies_save_binary())
	{
		ss << file_manager.get_base_filename() << ".0.obs.jcb";
		oe.to_binary(ss.str());
	}
	else
	{
		ss << file_manager.get_base_filename() << ".0.obs.csv";
		oe.to_csv(ss.str());
	}
	message(1, "saved initial obs ensemble to", ss.str());

	//save the 0th iter par and rei and well as the untagged par and rei
	save_base_real_par_rei(pest_scenario, dv, oe, output_file_writer, file_manager, iter);
	save_base_real_par_rei(pest_scenario, dv, oe, output_file_writer, file_manager, -1);


	performance_log->log_event("calc pre-drop phi");
	//initialize the phi handler
	ph = L2PhiHandler(&pest_scenario, &file_manager, &oe_base, &dv_base, &parcov);

	if (ph.get_lt_obs_names().size() > 0)
	{
		message(1, "less_than inequality defined for observations: ", ph.get_lt_obs_names().size());
	}
	if (ph.get_gt_obs_names().size())
	{
		message(1, "greater_than inequality defined for observations: ", ph.get_gt_obs_names().size());
	}

	ph.update(oe, dv);
	message(0, "pre-drop initial phi summary");
	ph.report(true);
	drop_bad_phi(dv, oe);
	if (oe.shape().first == 0)
	{
		throw_sqp_error(string("all realizations dropped as 'bad'"));
	}
	if (oe.shape().first <= error_min_reals)
	{
		message(0, "too few active realizations:", oe.shape().first);
		message(1, "need at least ", error_min_reals);
		throw_sqp_error(string("too few active realizations, cannot continue"));
	}
	if (oe.shape().first < warn_min_reals)
	{
		ss.str("");
		ss << "WARNING: less than " << warn_min_reals << " active realizations...might not be enough";
		string s = ss.str();
		message(0, s);
	}

	pcs = ParChangeSummarizer(&dv_base, &file_manager,&output_file_writer);
	

	message(0, "initialization complete");
}


void SeqQuadProgram::drop_bad_phi(ParameterEnsemble &_pe, ObservationEnsemble &_oe, bool is_subset)
{
	//don't use this assert because _pe maybe full size, but _oe might be subset size
	if (!is_subset)
		if (_pe.shape().first != _oe.shape().first)
			throw_sqp_error("SeqQuadProgram::drop_bad_phi() error: _pe != _oe and not subset");
		
	double bad_phi = pest_scenario.get_pestpp_options().get_ies_bad_phi();
	double bad_phi_sigma = pest_scenario.get_pestpp_options().get_ies_bad_phi_sigma();
	vector<int> idxs = ph.get_idxs_greater_than(bad_phi,bad_phi_sigma, _oe);

	if (pest_scenario.get_pestpp_options().get_ies_debug_bad_phi())
		idxs.push_back(0);
	
	if (idxs.size() > 0)
	{

		message(0, "dropping realizations as bad: ", idxs.size());

		vector<string> par_real_names = _pe.get_real_names(), obs_real_names = _oe.get_real_names();
		stringstream ss;
		string pname;
		string oname;

		int pidx;
		vector<string> full_onames, full_pnames;
		// if a subset drop, then use the full oe index, otherwise, just use _oe index
		/*if (_oe.shape().first != _pe.shape().first)
		{
			full_onames = oe.get_real_names();
		}
		else
		{
			full_onames = _oe.get_real_names();
		}*/
		full_onames = oe.get_real_names();
		full_pnames = dv.get_real_names();
		vector<string> pdrop, odrop;
		for (auto i : idxs)
		{
			oname = obs_real_names[i];
			
			if (is_subset)
			{
				pidx = find(full_onames.begin(), full_onames.end(), oname) - full_onames.begin();
				if (find(subset_idxs.begin(), subset_idxs.end(), pidx) == subset_idxs.end())
				{
					ss.str("");
					ss << "drop_bad_phi() error: idx " << pidx << " not found in subset_idxs";
					throw_sqp_error(ss.str());
				}
				pname = full_pnames[pidx];
			}
			else
			{
				pidx = i;
				pname = par_real_names[pidx];
			}
			ss << pname << " : " << obs_real_names[i] << " , ";
			pdrop.push_back(pname);
			odrop.push_back(obs_real_names[i]);
		}

		string s = "dropping par:obs realizations: "+ ss.str();
		message(1, s);
		try
		{
			_pe.drop_rows(pdrop);
			_oe.drop_rows(odrop);
		}
		catch (const exception &e)
		{
			stringstream ss;
			ss << "drop_bad_phi() error : " << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("drop_bad_phi() error"));
		}
	}
}

void SeqQuadProgram::save_mat(string prefix, Eigen::MatrixXd &mat)
{
	stringstream ss;
	ss << iter << '.' << prefix;
	try
	{
		ofstream &f = file_manager.open_ofile_ext(ss.str());
		f << mat << endl;
		f.close();
		file_manager.close_file(ss.str());
	}
	catch (...)
	{
		message(1, "error saving matrix", ss.str());
	}
}


void SeqQuadProgram::iterate_2_solution()
{
	stringstream ss;
	ofstream &frec = file_manager.rec_ofstream();
	
	bool accept;
	for (int i = 0; i < pest_scenario.get_control_info().noptmax; i++)
	{
		iter++;
		message(0, "starting solve for iteration:", iter);
		ss.str("");
		ss << "starting solve for iteration: " << iter;
		performance_log->log_event(ss.str());
		accept = solve_new();
		report_and_save();
		ph.update(oe,dv);
		last_best_mean = ph.get_mean(L2PhiHandler::phiType::COMPOSITE);
		last_best_std = ph.get_std(L2PhiHandler::phiType::COMPOSITE);
		ph.report(true);
		ph.write(iter, run_mgr_ptr->get_total_runs());
		if (pest_scenario.get_pestpp_options().get_ies_save_rescov())
			ph.save_residual_cov(oe,iter);
		ss.str("");
		ss << file_manager.get_base_filename() << "." << iter << ".pcs.csv";
		pcs.summarize(dv,iter,ss.str());
			
		if (should_terminate())
			break;
	}
}

bool SeqQuadProgram::should_terminate()
{
	//todo: work out some SQP-based criteria here...
	//double phiredstp = pest_scenario.get_control_info().phiredstp;
	//int nphistp = pest_scenario.get_control_info().nphistp;
	//int nphinored = pest_scenario.get_control_info().nphinored;
	//bool phiredstp_sat = false, nphinored_sat = false, consec_sat = false;
	//double phi, ratio;
	//int count = 0;
	//int nphired = 0;

	//message(0, "phi-based termination criteria check");
	//message(1, "phiredstp: ", phiredstp);
	//message(1, "nphistp: ", nphistp);
	//message(1, "nphinored (also used for consecutive bad lambda cycles): ", nphinored);
	//if (best_mean_phis.size() > 0)
	//{
	//	vector<double>::iterator idx = min_element(best_mean_phis.begin(), best_mean_phis.end());
	//	nphired = (best_mean_phis.end() - idx) - 1;
	//	best_phi_yet = best_mean_phis[idx - best_mean_phis.begin()];// *pest_scenario.get_pestpp_options().get_ies_accept_phi_fac();
	//	message(1, "best mean phi sequence: ", best_mean_phis);
	//	message(1, "best phi yet: ", best_phi_yet);
	//}
	//message(1, "number of consecutive bad lambda testing cycles: ", consec_bad_lambda_cycles);
	//if (consec_bad_lambda_cycles >= nphinored)
	//{
	//	message(1, "number of consecutive bad lambda testing cycles > nphinored");
	//	consec_sat = true;
	//}

	//for (auto &phi : best_mean_phis)
	//{
	//	ratio = (phi - best_phi_yet) / phi;
	//	if (ratio <= phiredstp)
	//		count++;
	//}
	//message(1, "number of iterations satisfying phiredstp criteria: ", count);
	//if (count >= nphistp)
	//{
	//	message(1, "number iterations satisfying phiredstp criteria > nphistp");
	//	phiredstp_sat = true;
	//}

	//message(1, "number of iterations since best yet mean phi: ", nphired);
	//if (nphired >= nphinored)
	//{
	//	message(1, "number of iterations since best yet mean phi > nphinored");
	//	nphinored_sat = true;
	//}

	//if ((nphinored_sat) || (phiredstp_sat) || (consec_sat))
	//{
	//	message(1, "phi-based termination criteria satisfied, all done");
	//	return true;
	//}
	return false;
}







void SeqQuadProgram::update_reals_by_phi(ParameterEnsemble &_pe, ObservationEnsemble &_oe)
{

	vector<string> oe_names = _oe.get_real_names();
	vector<string> pe_names = _pe.get_real_names();
	vector<string> oe_base_names = oe.get_real_names();
	vector<string> pe_base_names = dv.get_real_names();

	//if (pe_names.size() != oe_base_names.size())
	//	throw runtime_error("SeqQuadProgram::update_reals_by_phi() error: pe_names != oe_base_names");
	map<string, int> oe_name_to_idx;
	map<int,string> pe_idx_to_name;

	for (int i = 0; i < oe_base_names.size(); i++)
		oe_name_to_idx[oe_base_names[i]] = i;
	
	for (int i = 0; i < pe_base_names.size(); i++)
		pe_idx_to_name[i] = pe_base_names[i];
	//store map of current phi values
	ph.update(oe, dv);
	L2PhiHandler::phiType pt = L2PhiHandler::phiType::COMPOSITE;
	map<string, double> *phi_map = ph.get_phi_map(pt);
	map<string, double> cur_phi_map;
	for (auto p : *phi_map)
		cur_phi_map[p.first] = p.second;

	//now get a phi map of the new phi values
	ph.update(_oe, _pe);
	phi_map = ph.get_phi_map(pt);
	
	double acc_fac = pest_scenario.get_pestpp_options().get_ies_accept_phi_fac();
	double cur_phi, new_phi;
	string oname, pname;
	Eigen::VectorXd real;
	stringstream ss;
	for (int i=0;i<_oe.shape().first;i++)
	{
		oname = oe_names[i];
		new_phi = phi_map->at(oname);
		cur_phi = cur_phi_map.at(oname);
		if (new_phi < cur_phi * acc_fac)
		{
			//pname = pe_names[i];
			//pname = pe_names[oe_name_to_idx[oname]];
			pname = pe_idx_to_name[oe_name_to_idx[oname]];
			if (find(pe_names.begin(), pe_names.end(), pname) == pe_names.end())
				throw runtime_error("SeqQuadProgram::update_reals_by_phi() error: pname not in pe_names: " + pname);
			ss.str("");
			ss << "updating dv:oe real =" << pname << ":" << oname << ", current phi: new phi  =" << cur_phi << ":" << new_phi;
			message(3, ss.str());
			
			real = _pe.get_real_vector(pname);
			dv.update_real_ip(pname, real);
			real = _oe.get_real_vector(oname);
			oe.update_real_ip(oname, real);
		}
	}
	ph.update(oe, dv);

}

ParameterEnsemble SeqQuadProgram::fancy_solve_routine(double scale_val)
{
	ParameterEnsemble dv_candidate = dv; //copy
	//lots of fancy maths here...
	return dv_candidate;
}

bool SeqQuadProgram::solve_new()
{
	stringstream ss;
	ofstream &frec = file_manager.rec_ofstream();
	if (dv.shape().first <= error_min_reals)
	{
		message(0, "too few active realizations:", oe.shape().first);
		message(1, "need at least ", error_min_reals);
		throw_sqp_error(string("too few active realizations, cannot continue"));
	}
	if (dv.shape().first < warn_min_reals)
	{
		ss.str("");
		ss << "WARNING: less than " << warn_min_reals << " active realizations...might not be enough";
		string s = ss.str();
		message(1, s);
	}

	if ((use_subset) && (subset_size > dv.shape().first))
	{
		ss.str("");
		ss << "++ies_subset size (" << subset_size << ") greater than ensemble size (" << dv.shape().first << ")";
		frec << "  ---  " << ss.str() << endl;
		cout << "  ---  " << ss.str() << endl;
		frec << "  ...reducing ++ies_subset_size to " << dv.shape().first << endl;
		cout << "  ...reducing ++ies_subset_size to " << dv.shape().first << endl;
		subset_size = dv.shape().first;
	}

	dv.transform_ip(ParameterEnsemble::transStatus::NUM);

	//vector to store candidate upgrade ensembles
	vector<ParameterEnsemble> dv_candidates;
	//the current candidate
	ParameterEnsemble dv_candidate;
	
	//backtracking/line search factors
	//TODO: make this a ++ arg or tunable or something clever
	vector<double> scale_vals{ 0.1,0.5,1.0 };
	
	for (auto &scale_val : scale_vals)
	{
		ss.str("");
		ss << "starting calcs for scaling factor" << scale_val;
		message(1, "starting lambda calcs for scaling factor", scale_val);
		message(2, "see .log file for more details");
		
		dv_candidate = fancy_solve_routine(scale_val);

		dv_candidates.push_back(dv_candidate);
		//TODO: add sqp option to save candidates, but for now, just piggy back on 
		//ies option
		if (!pest_scenario.get_pestpp_options().get_ies_save_lambda_en())
			continue;
		ss.str("");
		ss << file_manager.get_base_filename() << "." << iter << "." << scale_val << ".scale.par";

		if (pest_scenario.get_pestpp_options().get_ies_save_binary())
		{
			ss << ".jcb";
			dv_candidate.to_binary(ss.str());
		}
		else
		{
			ss << ".csv";
			dv_candidate.to_csv(ss.str());
		}
		frec << "scale value " << scale_val << " dv ensemble saved to " << ss.str() << endl;

		

		ss.str("");
		message(1, "finished calcs for scaling factor:", scale_val);

	}
	
	if (pest_scenario.get_pestpp_options().get_ies_debug_upgrade_only())
	{
		message(0, "ies_debug_upgrade_only is true, exiting");
		throw_sqp_error("ies_debug_upgrade_only is true, exiting");
	}

	vector<map<int, int>> real_run_ids_lams;
	int best_idx = -1;
	double best_mean = 1.0e+30, best_std = 1.0e+30;
	double mean, std;

	message(0, "running candidate decision variable ensembles");
	vector<ObservationEnsemble> oe_lams = run_candidate_ensembles(dv_candidates, scale_vals);

	message(0, "evaluting lambda ensembles");
	message(1, "last mean: ", last_best_mean);
	message(1, "last stdev: ", last_best_std);

	ObservationEnsemble oe_scale_best;
	bool echo = false;
	if (verbose_level > 1)
		echo = true;
	for (int i = 0; i<dv_candidates.size(); i++)
	{
		if (oe_lams[i].shape().first == 0)
			continue;
		if (pest_scenario.get_pestpp_options().get_ies_save_lambda_en())
			
		{
			ss.str("");
			ss << file_manager.get_base_filename() << "." << iter << "." << scale_vals[i] << ".scale.obs";

			if (pest_scenario.get_pestpp_options().get_ies_save_binary())
			{
				ss << ".jcb";
				oe_lams[i].to_binary(ss.str());
			}
			else
			{
				ss << ".csv";
				oe_lams[i].to_csv(ss.str());
			}
			frec << "scale value " << scale_vals[i] << " observation scale value ensemble saved to " << ss.str() << endl;

		}
		drop_bad_phi(dv_candidates[i], oe_lams[i], true);
		if (oe_lams[i].shape().first == 0)
		{
			message(1, "all realizations dropped as 'bad' for scale value: ", scale_vals[i]);
			continue;
		}
		
		ph.update(oe_lams[i], dv_candidates[i]);
		
		message(0, "phi summary for scale value: ", scale_vals[i]);
		ph.report(echo);
		
		mean = ph.get_mean(L2PhiHandler::phiType::COMPOSITE);
		std = ph.get_std(L2PhiHandler::phiType::COMPOSITE);
		if (mean < best_mean)
		{
			oe_scale_best = oe_lams[i];
			best_mean = mean;
			best_std = std;
			best_idx = i;
		}
	}
	if (best_idx == -1)
	{
		message(0, "WARNING:  unsuccessful lambda testing, resetting lambda to 10000.0");
		last_best_lam = 10000.0;
		return false;

	}
	double acc_fac = pest_scenario.get_pestpp_options().get_ies_accept_phi_fac();
	double lam_inc = pest_scenario.get_pestpp_options().get_ies_lambda_inc_fac();
	double lam_dec = pest_scenario.get_pestpp_options().get_ies_lambda_dec_fac();
	

	//subset stuff here
	if ((best_idx != -1) && (use_subset) && (subset_size < dv.shape().first))
	{

		double acc_phi = last_best_mean * acc_fac;
		
		if (pest_scenario.get_pestpp_options().get_ies_debug_high_subset_phi())
		{
			cout << "ies_debug_high_subset_phi active" << endl;
			best_mean = acc_phi + 1.0;
		}

		if (best_mean > acc_phi)
		{
			//ph.update(oe_lams[best_idx],pe_lams[best_idx]);
			
			//TODO: think about what bad sqp cycles look like and how to handle...
			/*ss.str("");
			ss << "best subset mean phi  (" << best_mean << ") greater than acceptable phi : " << acc_phi;
			string m = ss.str();
			message(0, m);
			message(1, "abandoning current lambda ensembles, increasing lambda to ", new_lam);
			message(1, "updating realizations with reduced phi");
			update_reals_by_phi(pe_lams[best_idx], oe_lams[best_idx]);
			message(1, "returing to lambda calculations...");
			return false;*/
		}

		//release the memory of the unneeded pe_lams
		for (int i = 0; i < dv_candidates.size(); i++)
		{
			if (i == best_idx)
				continue;
			dv_candidates[i] = ParameterEnsemble();
		}
		//need to work out which par and obs en real names to run - some may have failed during subset testing...
		ObservationEnsemble remaining_oe_lam = oe;//copy
		ParameterEnsemble remaining_pe_lam = dv_candidates[best_idx];
		vector<string> pe_keep_names, oe_keep_names;
		vector<string> pe_names = dv.get_real_names(), oe_names = oe.get_real_names();

		vector<string> org_pe_idxs,org_oe_idxs;
		set<string> ssub;
		for (auto &i : subset_idxs)
			ssub.emplace(pe_names[i]);
		for (int i=0;i<pe_names.size();i++)
			if (ssub.find(pe_names[i]) == ssub.end())
			{
				pe_keep_names.push_back(pe_names[i]);
				//oe_keep_names.push_back(oe_names[i]);
			}
		ssub.clear();
		for (auto &i : subset_idxs)
			ssub.emplace(oe_names[i]);
		for (int i = 0; i<oe_names.size(); i++)
			if (ssub.find(oe_names[i]) == ssub.end())
			{
				oe_keep_names.push_back(oe_names[i]);
			}
		message(0, "phi summary for best scale value: ", scale_vals[best_idx]);
		ph.update(oe_lams[best_idx], dv_candidates[best_idx]);
		ph.report(true);
		message(0, "running remaining realizations for best scale value:", scale_vals[best_idx]);

		//pe_keep_names and oe_keep_names are names of the remaining reals to eval
		performance_log->log_event("dropping subset idxs from remaining_pe_lam");
		remaining_pe_lam.keep_rows(pe_keep_names);
		performance_log->log_event("dropping subset idxs from remaining_oe_lam");
		remaining_oe_lam.keep_rows(oe_keep_names);
		//save these names for later
		org_pe_idxs = remaining_pe_lam.get_real_names();
		org_oe_idxs = remaining_oe_lam.get_real_names();
		///run
		vector<int> fails = run_ensemble(remaining_pe_lam, remaining_oe_lam);

		//for testing
		if (pest_scenario.get_pestpp_options().get_ies_debug_fail_remainder())
			fails.push_back(0);

		//if any of the remaining runs failed
		if (fails.size() == org_pe_idxs.size())
			throw_sqp_error(string("all remaining realizations failed...something is prob wrong"));
		if (fails.size() > 0)
		{

			vector<string> new_pe_idxs, new_oe_idxs;
			vector<int>::iterator start = fails.begin(), end = fails.end();
			stringstream ss;
			ss << "the following par:obs realizations failed during evaluation of the remaining ensemble: ";
			for (int i = 0; i < org_pe_idxs.size(); i++)
				if (find(start, end, i) == end)
				{
					new_pe_idxs.push_back(org_pe_idxs[i]);
					new_oe_idxs.push_back(org_oe_idxs[i]);
				}
				else
				{
					ss << org_pe_idxs[i] << ":" << org_oe_idxs[i] << " , ";
				}
			string s = ss.str();
			message(1, s);
			remaining_oe_lam.keep_rows(new_oe_idxs);
			remaining_pe_lam.keep_rows(new_pe_idxs);

		}
		//drop the remaining runs from the par en then append the remaining par runs (in case some failed)
		performance_log->log_event("assembling ensembles");
		dv_candidates[best_idx].drop_rows(pe_keep_names);
		dv_candidates[best_idx].append_other_rows(remaining_pe_lam);
		//append the remaining obs en
		oe_scale_best.append_other_rows(remaining_oe_lam);
		assert(dv_candidates[best_idx].shape().first == oe_scale_best.shape().first);
		drop_bad_phi(dv_candidates[best_idx], oe_scale_best);
		if (oe_scale_best.shape().first == 0)
		{
			throw_sqp_error(string("all realization dropped after finishing subset runs...something might be wrong..."));
		}
		performance_log->log_event("updating phi");
		ph.update(oe_scale_best, dv_candidates[best_idx]);
		best_mean = ph.get_mean(L2PhiHandler::phiType::COMPOSITE);
		best_std = ph.get_std(L2PhiHandler::phiType::COMPOSITE);
		message(1, "phi summary for entire ensemble using scale value",scale_vals[best_idx]);
		ph.report(true);
	}
	else
	{
		ph.update(oe_scale_best, dv_candidates[best_idx]);
		best_mean = ph.get_mean(L2PhiHandler::phiType::COMPOSITE);
		best_std = ph.get_std(L2PhiHandler::phiType::COMPOSITE);
		
	}

	ph.update(oe_scale_best, dv_candidates[best_idx]);
	best_mean = ph.get_mean(L2PhiHandler::phiType::COMPOSITE);
	best_std = ph.get_std(L2PhiHandler::phiType::COMPOSITE);
	message(1, "last best mean phi * acceptable phi factor: ", last_best_mean * acc_fac);
	message(1, "current best mean phi: ", best_mean);

	if (pest_scenario.get_pestpp_options().get_ies_debug_high_upgrade_phi())
	{
		cout << "ies_debug_high_upgrade_phi active" << endl;
		best_mean = (last_best_mean * acc_fac) + 1.0;
	}

	//track this here for phi-based termination check
	best_mean_phis.push_back(best_mean);


	if (best_mean < last_best_mean * acc_fac)
	{
		message(0, "updating parameter ensemble");
		performance_log->log_event("updating parameter ensemble");
		last_best_mean = best_mean;

		dv = dv_candidates[best_idx];
		oe = oe_scale_best;
		/*if (best_std < last_best_std * acc_fac)
		{
			double new_lam = lam_vals[best_idx] * lam_dec;
			new_lam = (new_lam < lambda_min) ? lambda_min : new_lam;
			message(0, "updating lambda to ", new_lam);
			last_best_lam = new_lam;
		}
		else
		{
			message(0, "not updating lambda");
		}*/
		last_best_std = best_std;
	}

	else 
	{
		//message(0, "not updating parameter ensemble");
		message(0, "only updating realizations with reduced phi");
		update_reals_by_phi(dv_candidates[best_idx], oe_scale_best);
		ph.update(oe, dv);
		/*double new_lam = last_best_lam * lam_inc;
		new_lam = (new_lam > lambda_max) ? lambda_max : new_lam;
		message(0, "incresing lambda to: ", new_lam);
		last_best_lam = new_lam;*/
	}
	//report_and_save();
	return true;
}


void SeqQuadProgram::report_and_save()
{
	ofstream &frec = file_manager.rec_ofstream();
	frec << endl << "  ---  SeqQuadProgram iteration " << iter << " report  ---  " << endl;
	frec << "   number of active realizations:  " << dv.shape().first << endl;
	frec << "   number of model runs:           " << run_mgr_ptr->get_total_runs() << endl;

	cout << endl << "  ---  SeqQuadProgram iteration " << iter << " report  ---  " << endl;
	cout << "   number of active realizations:   " << dv.shape().first << endl;
	cout << "   number of model runs:            " << run_mgr_ptr->get_total_runs() << endl;

	stringstream ss;
	if (pest_scenario.get_pestpp_options().get_ies_save_binary())
	{
		ss << file_manager.get_base_filename() << "." << iter << ".obs.jcb";
		oe.to_binary(ss.str());
	}
	else
	{
		ss << file_manager.get_base_filename() << "." << iter << ".obs.csv";
		oe.to_csv(ss.str());
	}
	frec << "      current obs ensemble saved to " << ss.str() << endl;
	cout << "      current obs ensemble saved to " << ss.str() << endl;
	ss.str("");
	if (pest_scenario.get_pestpp_options().get_ies_save_binary())
	{
		ss << file_manager.get_base_filename() << "." << iter << ".par.jcb";
		dv.to_binary(ss.str());
	}
	else
	{
		ss << file_manager.get_base_filename() << "." << iter << ".par.csv";
		dv.to_csv(ss.str());
	}
	save_base_real_par_rei(pest_scenario, dv, oe, output_file_writer, file_manager, iter);
	save_base_real_par_rei(pest_scenario, dv, oe, output_file_writer, file_manager, -1);
	//ss << file_manager.get_base_filename() << "." << iter << ".par.csv";
	//dv.to_csv(ss.str());
	frec << "      current par ensemble saved to " << ss.str() << endl;
	cout << "      current par ensemble saved to " << ss.str() << endl;

	

}


void SeqQuadProgram::set_subset_idx(int size)
{
	//map<int,int> subset_idx_map;
	subset_idxs.clear();
	int nreal_subset = pest_scenario.get_pestpp_options().get_ies_subset_size();
	if ((!use_subset) || (nreal_subset >= size))
	{
		for (int i = 0; i < size; i++)
			subset_idxs.push_back(i);
		return;
	}
	vector<string> pe_names = dv.get_real_names();

	vector<string>::iterator bidx = find(pe_names.begin(), pe_names.end(), base_name);
	if (bidx != pe_names.end())
	{

		subset_idxs.push_back(bidx - pe_names.begin());
	}
	//int size = dv.shape().first;
	string how = pest_scenario.get_pestpp_options().get_ies_subset_how();
	if (how == "FIRST")
	{
		for (int i = 0; i < size; i++)
		{
			if (subset_idxs.size() >= nreal_subset)
				break;
			if (find(subset_idxs.begin(), subset_idxs.end(), i) != subset_idxs.end())
				continue;

			subset_idxs.push_back(i);

		}

	}
	else if (how == "LAST")
	{

		for (int i = size-1; i >= 0; i--)
		{
			if (subset_idxs.size() >= nreal_subset)
				break;
			if (find(subset_idxs.begin(), subset_idxs.end(), i) != subset_idxs.end())
				continue;

			subset_idxs.push_back(i);

		}

	}

	else if (how == "RANDOM")
	{
		std::uniform_int_distribution<int> uni(0, size-1);
		int idx;
		for (int i = 0; i < 10000000; i++)
		{
			if (subset_idxs.size() >= nreal_subset)
				break;
			idx = uni(subset_rand_gen);
			if (find(subset_idxs.begin(), subset_idxs.end(), idx) != subset_idxs.end())
				continue;
			subset_idxs.push_back(idx);
		}
		if (subset_idxs.size() != nreal_subset)
			throw_sqp_error("max iterations exceeded when trying to find random subset idxs");

	}
	else if (how == "PHI_BASED")
	{
		//sidx needs to be index of realization, not realization number
		vector<pair<double, int>> phis;
		//vector<int> sidx;
		int step;
		int idx;
		L2PhiHandler::phiType pt = L2PhiHandler::phiType::COMPOSITE;
		map<string, double>* phi_map = ph.get_phi_map(pt);
		map<string, double>::iterator pi = phi_map->begin(), end = phi_map->end();

		int i = 0;
		for (; pi != end; ++pi)
		{
			phis.push_back(make_pair(pi->second, i)); //phival,idx?
			++i;
		}
		sort(phis.begin(), phis.end());

		//include idx for lowest and highest phi reals
		if (subset_idxs.size() < nreal_subset)
		{
			for (auto phi : phis)
			{
				if (find(subset_idxs.begin(), subset_idxs.end(), phi.second) == subset_idxs.end())
				{
					subset_idxs.push_back(phi.second);
					break;
				}
			}
		}
		if (subset_idxs.size() < nreal_subset)
		{
			for (int i = phis.size() - 1; i >= 0; i--)
			{
				if (find(subset_idxs.begin(), subset_idxs.end(), phis[i].second) == subset_idxs.end())
				{
					subset_idxs.push_back(phis[i].second);
					break;
				}
			}
		}


		step = (phis.size()-1) / nreal_subset;
		//cout << step << endl;
		//cout << (phis.size() - 1) << endl;
		for (i = 1; i < nreal_subset; ++i)
		{
			//add higher phis first
			idx = phis.size() - (i * step);
			if ((subset_idxs.size() < nreal_subset) && (find(subset_idxs.begin(), subset_idxs.end(), phis[idx].second) == subset_idxs.end()))
			{
				subset_idxs.push_back(phis[idx].second);
				//cout << i << endl;
				//cout << idx << endl;
				//cout << phis[idx].first << endl;
				//cout << phis[idx].second << endl;
			}
		}
	}
	else
	{
		//throw runtime_error("unkonwn 'subset_how'");
		throw_sqp_error("unknown 'subset_how'");
	}
	stringstream ss;
	for (auto i : subset_idxs)
		ss << i << ":" << pe_names[i] << ", ";
	message(1,"subset idx:dv real name: ",ss.str());
	return;
	//return subset_idx_map;
}

vector<ObservationEnsemble> SeqQuadProgram::run_candidate_ensembles(vector<ParameterEnsemble>& dv_candidates, vector<double> &scale_vals)
{
	ofstream &frec = file_manager.rec_ofstream();
	stringstream ss;
	ss << "queuing " << dv_candidates.size() << " ensembles";
	performance_log->log_event(ss.str());
	run_mgr_ptr->reinitialize();
	
	set_subset_idx(dv_candidates[0].shape().first);
	vector<map<int, int>> real_run_ids_vec;
	//ParameterEnsemble pe_lam;
	//for (int i=0;i<pe_lams.size();i++)
	for (auto &dv_candidate : dv_candidates)
	{
		try
		{
			real_run_ids_vec.push_back(dv_candidate.add_runs(run_mgr_ptr,subset_idxs));
		}
		catch (const exception &e)
		{
			stringstream ss;
			ss << "run_ensemble() error queueing runs: " << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			throw_sqp_error(string("run_ensembles() error queueing runs"));
		}
	}
	performance_log->log_event("making runs");
	try
	{

		run_mgr_ptr->run();
	}
	catch (const exception &e)
	{
		stringstream ss;
		ss << "error running ensembles: " << e.what();
		throw_sqp_error(ss.str());
	}
	catch (...)
	{
		throw_sqp_error(string("error running ensembles"));
	}

	performance_log->log_event("processing runs");
	vector<int> failed_real_indices;
	vector<ObservationEnsemble> obs_lams;
	//ObservationEnsemble _oe = oe;//copy
	//if (subset_size < pe_lams[0].shape().first)
	//	_oe.keep_rows(subset_idxs);
	map<int, int> real_run_ids;
	//for (auto &real_run_ids : real_run_ids_vec)
	for (int i=0;i<dv_candidates.size();i++)
	{
		ObservationEnsemble _oe = oe;//copy
		//vector<double> rep_vals{ lam_vals[i],scale_vals[i] };
		real_run_ids = real_run_ids_vec[i];
		//if using subset, reset the real_idx in real_run_ids to be just simple counter
		//if (subset_size < pe_lams[0].shape().first)
		if ((use_subset) && (subset_size < dv_candidates[i].shape().first))
		{
			_oe.keep_rows(subset_idxs);
			int ireal = 0;
			map<int, int> temp;
			for (auto &rri : real_run_ids)
			{
				temp[ireal] = rri.second;
				ireal++;
			}

			real_run_ids = temp;
		}

		try
		{
			failed_real_indices = _oe.update_from_runs(real_run_ids, run_mgr_ptr);
		}
		catch (const exception &e)
		{
			stringstream ss;
			ss << "error processing runs for scale value: " << scale_vals[i] << ':' << e.what();
			throw_sqp_error(ss.str());
		}
		catch (...)
		{
			stringstream ss;
			ss << "error processing runs for scale value: " << scale_vals[i];
			throw_sqp_error(ss.str());
		}

		if (pest_scenario.get_pestpp_options().get_ies_debug_fail_subset())
			failed_real_indices.push_back(real_run_ids.size()-1);

		if (failed_real_indices.size() > 0)
		{
			stringstream ss;
			vector<string> par_real_names = dv.get_real_names();
			vector<string> obs_real_names = oe.get_real_names();
			vector<string> failed_par_names, failed_obs_names;
			string oname, pname;
			ss << "the following dv:obs realization runs failed for scale value " << scale_vals[i] << "-->";
			for (auto &i : failed_real_indices)
			{
				pname = par_real_names[subset_idxs[i]];
				oname = obs_real_names[subset_idxs[i]];
				failed_par_names.push_back(pname);
				failed_obs_names.push_back(oname);
				ss << pname << ":" << oname << ',';
			}
			string s = ss.str();
			message(1,s);
			if (failed_real_indices.size() == _oe.shape().first)
			{
				message(0, "WARNING: all realizations failed for scale value :", scale_vals[i]);
				_oe = ObservationEnsemble();

			}
			else
			{
				performance_log->log_event("dropping failed realizations");
				//_oe.drop_rows(failed_real_indices);
				//pe_lams[i].drop_rows(failed_real_indices);
				_oe.drop_rows(failed_obs_names);
				dv_candidates[i].drop_rows(failed_par_names);
			}

		}
		obs_lams.push_back(_oe);
	}
	return obs_lams;
}


vector<int> SeqQuadProgram::run_ensemble(ParameterEnsemble &_pe, ObservationEnsemble &_oe, const vector<int> &real_idxs)
{
	stringstream ss;
	ss << "queuing " << _pe.shape().first << " runs";
	performance_log->log_event(ss.str());
	run_mgr_ptr->reinitialize();
	map<int, int> real_run_ids;
	try
	{
		real_run_ids = _pe.add_runs(run_mgr_ptr,real_idxs);
	}
	catch (const exception &e)
	{
		stringstream ss;
		ss << "run_ensemble() error queueing runs: " << e.what();
		throw_sqp_error(ss.str());
	}
	catch (...)
	{
		throw_sqp_error(string("run_ensemble() error queueing runs"));
	}
	performance_log->log_event("making runs");
	try
	{
		run_mgr_ptr->run();
	}
	catch (const exception &e)
	{
		stringstream ss;
		ss << "error running ensemble: " << e.what();
		throw_sqp_error(ss.str());
	}
	catch (...)
	{
		throw_sqp_error(string("error running ensemble"));
	}

	performance_log->log_event("processing runs");
	if (real_idxs.size() > 0)
	{
		_oe.keep_rows(real_idxs);
	}
	vector<int> failed_real_indices;
	try
	{
		failed_real_indices = _oe.update_from_runs(real_run_ids,run_mgr_ptr);
	}
	catch (const exception &e)
	{
		stringstream ss;
		ss << "error processing runs: " << e.what();
		throw_sqp_error(ss.str());
	}
	catch (...)
	{
		throw_sqp_error(string("error processing runs"));
	}
	//for testing
	//failed_real_indices.push_back(0);

	if (failed_real_indices.size() > 0)
	{
		stringstream ss;
		vector<string> par_real_names = _pe.get_real_names();
		vector<string> obs_real_names = _oe.get_real_names();
		ss << "the following par:obs realization runs failed: ";
		for (auto &i : failed_real_indices)
		{
			ss << par_real_names[i] << ":" << obs_real_names[i] << ',';
		}
		performance_log->log_event(ss.str());
		message(1, "failed realizations: ", failed_real_indices.size());
		string s = ss.str();
		message(1, s);
		performance_log->log_event("dropping failed realizations");
		_pe.drop_rows(failed_real_indices);
		_oe.drop_rows(failed_real_indices);
	}
	return failed_real_indices;
}


void SeqQuadProgram::finalize()
{

}
