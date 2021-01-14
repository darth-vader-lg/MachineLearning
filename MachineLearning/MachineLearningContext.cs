using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;

namespace MachineLearning
{
   /// <summary>
   /// Contesto di machine learning
   /// </summary>
   [Serializable]
   public class MachineLearningContext : IMachineLearningContextProvider
   {
      #region Fields
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      [NonSerialized]
      private MLContext net;
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private readonly int? seed;
      #endregion
      #region Properties
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      MachineLearningContext IMachineLearningContextProvider.ML => this;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      public MLContext NET { get => net ??= new MLContext(seed); private set { net = value; } }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme per le operazioni random</param>
      public MachineLearningContext(int? seed = null) => this.seed = seed;
      /// <summary>
      /// Verifica che un provider di contesto sia valido per l'ML.NET
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void AssertMLNET(IMachineLearningContextProvider provider, string name)
      {
         Contracts.AssertValue(provider, name);
         Contracts.AssertValue(provider.ML, $"{name}.{nameof(IMachineLearningContextProvider.ML)}");
         Contracts.AssertValue(provider.ML.NET, $"{name}.{nameof(IMachineLearningContextProvider.ML.NET)}");
      }
      /// <summary>
      /// Verifica che un provider di contesto sia valido per l'ML.NET
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void CheckMLNET(IMachineLearningContextProvider provider, string name)
      {
         Contracts.CheckValue(provider, name);
         Contracts.CheckValue(provider.ML, $"{name}.{nameof(IMachineLearningContextProvider.ML)}");
         Contracts.CheckValue(provider.ML.NET, $"{name}.{nameof(IMachineLearningContextProvider.ML.NET)}");
      }
      #endregion
   }
}
