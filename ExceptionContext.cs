using MachineLearning.Model;
using Microsoft.ML.Runtime;
using System;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per gli oggetti con contesto di gestione eccezioni
   /// </summary>
   [Serializable]
   public abstract class ExceptionContext : IExceptionContext
   {
      #region Properties
      /// <summary>
      /// Descrizione del contesto
      /// </summary>
      string IExceptionContext.ContextDescription => (this as IModelName)?.ModelName ?? GetExceptionContext().ContextDescription;
      #endregion
      #region Methods
      /// <summary>
      /// Funzione di ottenimento del contesto di eccezione
      /// </summary>
      /// <returns>Il contesto</returns>
      protected abstract IExceptionContext GetExceptionContext();
      /// <summary>
      /// Processa un'eccezione
      /// </summary>
      /// <typeparam name="TException">Tipo di eccezione</typeparam>
      /// <param name="ex">Eccezione</param>
      /// <returns>L'eccezione processata</returns>
      TException IExceptionContext.Process<TException>(TException ex) => GetExceptionContext().Process(ex);
      #endregion
   }
}
