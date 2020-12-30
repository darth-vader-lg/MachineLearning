using Microsoft.ML.Data;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i providers di opzioni di testo
   /// </summary>
   public interface ITextLoaderOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento dati testuali
      /// </summary>
      TextLoader.Options TextLoaderOptions { get; }
      #endregion
   }
}
