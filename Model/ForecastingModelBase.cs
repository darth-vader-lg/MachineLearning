using Microsoft.ML;
using System;
using TTrainers = MachineLearning.Trainers.ForecastingTrainers;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i classificatori binari
   /// </summary>
   [Serializable]
   public abstract class ForecastingModelBase : ModelBaseMLNet
   {
      #region Properties
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      public TTrainers Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider contesto di machine learning</param>
      public ForecastingModelBase(IContextProvider<MLContext> contextProvider = default) : base(contextProvider) =>
         Trainers = new TTrainers(this);
      #endregion
   }
}
