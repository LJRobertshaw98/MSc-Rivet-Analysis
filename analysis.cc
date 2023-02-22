#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/VisibleFinalState.hh"
#include "Rivet/Projections/InvisibleFinalState.hh"
#include "Rivet/Projections/DISFinalState.hh"
#include "Rivet/Projections/Thrust.hh"
#include "Rivet/Projections/Sphericity.hh"

namespace Rivet {


  /// @brief Add a short analysis description here
  class EP_H_BOOSTED : public Analysis {
  public:

    /// Constructor
    RIVET_DEFAULT_ANALYSIS_CTOR(EP_H_BOOSTED);

    /// @name Analysis methods
    ///@{

    /// Book histograms and initialise projections before the run
    void init() {

      const VisibleFinalState vfs;
      declare(vfs, "VFS_projection");

      // Book histograms
      const int nbins = 200.0;

      book(_thrust, "Thrust", nbins, 0.0, 1.0);
      book(_thrustMajor, "Thrust_Major", nbins, 0.0, 1.0);
      book(_thrustMinor, "Thrust_Minor", nbins, 0.0, 1.0);

      book(_sphericity, "Sphericity", nbins, 0.0, 1.0);
      book(_planarity, "Planarity", nbins, 0.0, 1.0);
      book(_aplanarity, "Aplanarity", nbins, 0.0, 1.0);

      book(_energy, "Energy_of_each_particle_in_VFS", nbins, 0.0, 100.0);
      book(_cfsCount, "Count_of_charged_particles_in_VFS", nbins, 0.0, 100.0);
      book(_nfsCount, "Count_of_neutral_particles_in_VFS", nbins, 0.0, 100.0);
      book(_cfsPt, "Transverse_momentum_of_each_charged_particle_in_VFS", nbins, 0.0, 50.0);
      book(_nfsPt, "Transverse_momentum_of_each_neutral_particle_in_VFS", nbins, 0.0, 50.0);
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      const Particles& vfs_particles = apply<VisibleFinalState>(event, "VFS_projection").particles();

      // Reconstruct Higgs four momentum from sum of all decay products as four momentum is conserved.
      FourMomentum higgsMom;
      for(auto p : vfs_particles){
        higgsMom += p.mom();
      }

      // Boost the final state particles into the Higgs COM frame.
      const LorentzTransform lt = LorentzTransform::mkFrameTransform(higgsMom);
      Particles boosted_vfs_particles;
      for(auto p : vfs_particles){
        boosted_vfs_particles += p.transformBy(lt);
      }

      // Call filler fn to fill histograms for various event shapes.
      filler(boosted_vfs_particles);
    }


    // Normalise histos etc... 
    void finalize() {

      double luminosity = 1 / attobarn;
      for(auto h : {_thrust, _thrustMajor, _thrustMinor, _sphericity, _planarity, _aplanarity, _energy, _cfsCount, _nfsCount, _cfsPt, _nfsPt}){
        scale(h, ( (crossSection() / sumOfWeights()) *luminosity ));   /*  sumOfWeights() / crossSection() = numEvents();  */
      }
    }


    void filler(const Particles& particles){
      Thrust thrust;
      thrust.calc(particles);
      _thrust->fill(thrust.thrust());
      _thrustMajor->fill(thrust.thrustMajor());
      _thrustMinor->fill(thrust.thrustMinor());

      Sphericity sphericity;
      sphericity.calc(particles);
      _sphericity->fill(sphericity.sphericity());
      _planarity->fill(sphericity.planarity());
      _aplanarity->fill(sphericity.aplanarity());

      Particles charged_fs;
      Particles neutral_fs;
      for(auto p : particles){
        _energy->fill(p.E());
        if(p.isCharged() == true){
          charged_fs += p;
          _cfsPt->fill(p.pt());
        }
        else{
          neutral_fs += p;
          _nfsPt->fill(p.pt());
        }
      }
      _cfsCount->fill(charged_fs.size());
      _nfsCount->fill(neutral_fs.size());
    }

    ///@}

    /// @name Histograms
    ///@{
    Histo1DPtr _thrust, _thrustMajor, _thrustMinor, _sphericity, _planarity, _aplanarity, _energy, _cfsCount, _nfsCount, _cfsPt, _nfsPt;
    ///@}
  };

  RIVET_DECLARE_PLUGIN(EP_H_BOOSTED);
}
