using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Selettore di surrogati dell'assembly
   /// </summary>
   public class SurrogateSelector : ISurrogateSelector
   {
      #region Fields
      /// <summary>
      /// Selettore di surrogati dell'assembly per la serializzazione
      /// </summary>
      private static readonly System.Runtime.Serialization.SurrogateSelector _assemblySurrogates;
      /// <summary>
      /// Selettore successivo
      /// </summary>
      private ISurrogateSelector _nextSelector;
      #endregion
      #region Properties
      /// <summary>
      /// Contesto di streaming
      /// </summary>
      public static StreamingContext StreamingContext { get; } = new StreamingContext(StreamingContextStates.All, new MachineLearningContext());
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore statico
      /// </summary>
      static SurrogateSelector()
      {
         _assemblySurrogates = new System.Runtime.Serialization.SurrogateSelector();
         try {
            AppContext.SetSwitch("Switch.System.Runtime.Serialization.SerializationGuard.AllowFileWrites", true);
            var surrogates = Assembly.GetExecutingAssembly().GetTypes()
               .Select(t => (Surrogate: t, Interface: t.GetInterface($"{nameof(ISerializationSurrogate)}`1")))
               .Where(t => t.Interface != null)
               .Select(t => (t.Surrogate, Type: t.Interface.GetGenericArguments()[0]));
            var alTypes = new HashSet<Type>();
            foreach (var s in surrogates) {
               if (alTypes.Contains(s.Type))
                  continue;
               alTypes.Add(s.Type);
               var surrogate = (ISerializationSurrogate)Assembly.GetExecutingAssembly().CreateInstance(s.Surrogate.ToString());
               _assemblySurrogates.AddSurrogate(s.Type, StreamingContext, surrogate);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            throw;
         }
      }
      /// <summary>
      /// Forza il prossimo selettore di surrogati
      /// </summary>
      /// <param name="selector">Il prossimo selettore</param>
      internal void SetNextSelector(ISurrogateSelector selector) => _nextSelector = selector;
      /// <summary>
      /// Accoda un selettore alla catena
      /// </summary>
      /// <param name="selector"></param>
      public virtual void ChainSelector(ISurrogateSelector selector)
      {
         ISurrogateSelector temp;
         ISurrogateSelector tempCurr;
         ISurrogateSelector tempPrev;
         ISurrogateSelector tempEnd;
         if (selector == null)
            throw new ArgumentNullException(nameof(selector));
         if (selector == this)
            throw new SerializationException("Duplicate selector");
         if (!HasCycle(selector))
            throw new ArgumentException("Surrogate cycle in argument", nameof(selector));
         tempCurr = selector.GetNextSelector();
         tempEnd = selector;
         while (tempCurr != null && tempCurr != this) {
            tempEnd = tempCurr;
            tempCurr = tempCurr.GetNextSelector();
         }
         if (tempCurr == this)
            throw new ArgumentException("Surrogate cycle in argument", nameof(selector));
         tempCurr = selector;
         tempPrev = selector;
         while (tempCurr != null) {
            if (tempCurr == tempEnd)
               tempCurr = GetNextSelector();
            else
               tempCurr = tempCurr.GetNextSelector();
            if (tempCurr == null)
               break;
            if (tempCurr == tempPrev)
               throw new ArgumentException("Surrogate cycle in argument", nameof(selector));
            if (tempCurr == tempEnd)
               tempCurr = GetNextSelector();
            else
               tempCurr = tempCurr.GetNextSelector();
            if (tempPrev == tempEnd)
               tempPrev = GetNextSelector();
            else
               tempPrev = tempPrev.GetNextSelector();
            if (tempCurr == tempPrev)
               throw new ArgumentException("Surrogate cycle in argument", nameof(selector));
         }
         temp = _nextSelector;
         _nextSelector = selector;
         if (temp != null)
            tempEnd.ChainSelector(temp);
      }
      /// <summary>
      /// Restituisce il prossimo selettore nella catena
      /// </summary>
      /// <returns>Il selettore successivo</returns>
      public ISurrogateSelector GetNextSelector() => _nextSelector;
      /// <summary>
      /// Restituisce un surrogato
      /// </summary>
      /// <param name="type">Tipo</param>
      /// <param name="context">Contesto</param>
      /// <param name="selector">Il selettore di surrogato</param>
      /// <returns>Il surrogato</returns>
      public ISerializationSurrogate GetSurrogate(Type type, StreamingContext context, out ISurrogateSelector selector) => _assemblySurrogates.GetSurrogate(type, context, out selector);
      /// <summary>
      /// Verifica l'esistenza di una catena chiusa di selettori
      /// </summary>
      /// <param name="selector">Selettore da verificare</param>
      /// <returns>true se ok</returns>
      private static bool HasCycle(ISurrogateSelector selector)
      {
         ISurrogateSelector head;
         ISurrogateSelector tail;
         head = selector;
         tail = selector;
         while (head != null) {
            head = head.GetNextSelector();
            if (head == null)
               return true;
            if (head == tail)
               return false;
            head = head.GetNextSelector();
            tail = tail.GetNextSelector();
            if (head == tail)
               return false;
         }
         return true;
      }
      #endregion
   }
}
