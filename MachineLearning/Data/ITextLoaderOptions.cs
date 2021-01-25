using Microsoft.ML.Data;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per i contenitori di opzioni di testo
   /// </summary>
   public interface ITextLoaderOptions
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento dati testuali
      /// </summary>
      TextLoader.Options TextLoaderOptions { get; }
      #endregion
   }
}
